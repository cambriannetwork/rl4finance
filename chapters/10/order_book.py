from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List

@dataclass(frozen=True)
class DollarsAndShares:
    dollars: float
    shares: int

PriceSizePairs = Sequence[DollarsAndShares]

@dataclass(frozen=True)
class OrderBook:
    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        return self.descending_bids[0].dollars

    def ask_price(self) -> float:
        return self.ascending_asks[0].dollars

    def mid_price(self) -> float:
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        return self.ascending_asks[-1].dollars - self.descending_bids[-1].dollars

    @staticmethod
    def eat_book(ps_pairs: PriceSizePairs, shares: int) -> Tuple[DollarsAndShares, PriceSizePairs]:
        """
        Simulate consuming shares from the book side (bids or asks) until the order is filled or liquidity runs out.
        Returns the total dollars spent/earned and the updated book.
        """
        rem_shares = shares
        total_dollars = 0.0

        for i, level in enumerate(ps_pairs):
            take = min(rem_shares, level.shares)
            total_dollars += take * level.dollars
            rem_shares -= take

            if rem_shares == 0:
                # Return remaining shares at this level + untouched levels
                new_level = []
                if level.shares > take:
                    new_level = [DollarsAndShares(level.dollars, level.shares - take)]
                return DollarsAndShares(total_dollars, shares), new_level + list(ps_pairs[i+1:])

        # Partial fill: used up all liquidity
        return DollarsAndShares(total_dollars, shares - rem_shares), []

    def _insert_order(self, book: List[DollarsAndShares], price: float, shares: int, ascending: bool) -> List[DollarsAndShares]:
        """Helper to insert remaining shares into bid or ask book at correct sorted position."""
        for i, level in enumerate(book):
            if (ascending and level.dollars >= price) or (not ascending and level.dollars <= price):
                if level.dollars == price:
                    book[i] = DollarsAndShares(price, level.shares + shares)
                else:
                    book.insert(i, DollarsAndShares(price, shares))
                return book
        book.append(DollarsAndShares(price, shares))
        return book

    def sell_limit_order(self, price: float, shares: int) -> Tuple[DollarsAndShares, OrderBook]:
        index = next((i for i, d_s in enumerate(self.descending_bids) if d_s.dollars < price), None)
        eligible = self.descending_bids if index is None else self.descending_bids[:index]
        remainder = [] if index is None else self.descending_bids[index:]

        filled, rem_bids = self.eat_book(eligible, shares)
        new_bids = list(rem_bids) + list(remainder)
        unfilled = shares - filled.shares

        if unfilled > 0:
            new_asks = self._insert_order(list(self.ascending_asks), price, unfilled, ascending=True)
            return filled, OrderBook(descending_bids=new_bids, ascending_asks=new_asks)
        return filled, replace(self, descending_bids=new_bids)

    def sell_market_order(self, shares: int) -> Tuple[DollarsAndShares, OrderBook]:
        filled, rem_bids = self.eat_book(self.descending_bids, shares)
        return filled, replace(self, descending_bids=rem_bids)

    def buy_limit_order(self, price: float, shares: int) -> Tuple[DollarsAndShares, OrderBook]:
        index = next((i for i, d_s in enumerate(self.ascending_asks) if d_s.dollars > price), None)
        eligible = self.ascending_asks if index is None else self.ascending_asks[:index]
        remainder = [] if index is None else self.ascending_asks[index:]

        filled, rem_asks = self.eat_book(eligible, shares)
        new_asks = list(rem_asks) + list(remainder)
        unfilled = shares - filled.shares

        if unfilled > 0:
            new_bids = self._insert_order(list(self.descending_bids), price, unfilled, ascending=False)
            return filled, OrderBook(descending_bids=new_bids, ascending_asks=new_asks)
        return filled, replace(self, ascending_asks=new_asks)

    def buy_market_order(self, shares: int) -> Tuple[DollarsAndShares, OrderBook]:
        filled, rem_asks = self.eat_book(self.ascending_asks, shares)
        return filled, replace(self, ascending_asks=rem_asks)

    def pretty_print_order_book(self) -> None:
        from pprint import pprint
        print("\nBids")
        pprint(self.descending_bids)
        print("\nAsks")
        pprint(self.ascending_asks)

    def display_order_book(self) -> None:
        import matplotlib.pyplot as plt

        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        if self.descending_bids:
            plt.bar(bid_prices, bid_shares, color='blue')

        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        if self.ascending_asks:
            plt.bar(ask_prices, ask_shares, color='red')

        all_prices = sorted(bid_prices + ask_prices)
        plt.xticks(all_prices, [str(x) for x in all_prices])
        plt.grid(axis='y')
        plt.xlabel("Prices")
        plt.ylabel("Number of Shares")
        plt.title("Order Book")
        plt.show()

if __name__ == '__main__':

    from numpy.random import poisson

    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    print("Sell Limit Order of (107, 40)")
    print()
    d_s1, ob1 = ob0.sell_limit_order(107, 40)
    proceeds1: float = d_s1.dollars
    shares_sold1: int = d_s1.shares
    print(f"Sales Proceeds = {proceeds1:.2f}, Shares Sold = {shares_sold1:d}")
    ob1.pretty_print_order_book()
    ob1.display_order_book()

    print("Sell Market Order of 120")
    print()
    d_s2, ob2 = ob1.sell_market_order(120)
    proceeds2: float = d_s2.dollars
    shares_sold2: int = d_s2.shares
    print(f"Sales Proceeds = {proceeds2:.2f}, Shares Sold = {shares_sold2:d}")
    ob2.pretty_print_order_book()
    ob2.display_order_book()

    print("Buy Limit Order of (100, 80)")
    print()
    d_s3, ob3 = ob2.buy_limit_order(100, 80)
    bill3: float = d_s3.dollars
    shares_bought3: int = d_s3.shares
    print(f"Purchase Bill = {bill3:.2f}, Shares Bought = {shares_bought3:d}")
    ob3.pretty_print_order_book()
    ob3.display_order_book()

    print("Sell Limit Order of (104, 60)")
    print()
    d_s4, ob4 = ob3.sell_limit_order(104, 60)
    proceeds4: float = d_s4.dollars
    shares_sold4: int = d_s4.shares
    print(f"Sales Proceeds = {proceeds4:.2f}, Shares Sold = {shares_sold4:d}")
    ob4.pretty_print_order_book()
    ob4.display_order_book()

    print("Buy Market Order of 150")
    print()
    d_s5, ob5 = ob4.buy_market_order(150)
    bill5: float = d_s5.dollars
    shares_bought5: int = d_s5.shares
    print(f"Purchase Bill = {bill5:.2f}, Shares Bought = {shares_bought5:d}")
    ob5.pretty_print_order_book()
    ob5.display_order_book()