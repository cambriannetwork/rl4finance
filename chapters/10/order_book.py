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
        import matplotlib.patches as patches
        
        # Tufte-style configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.linewidth': 0.5,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
            'axes.grid': False
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove top and right spines (Tufte principle)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Lighten remaining spines
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        
        # Use minimal, meaningful colors - dark gray instead of bright colors
        if self.descending_bids:
            bars_bid = ax.bar(bid_prices, bid_shares, color='#404040', alpha=0.7, 
                             width=0.8, edgecolor='none')
        
        if self.ascending_asks:
            bars_ask = ax.bar(ask_prices, ask_shares, color='#808080', alpha=0.7, 
                             width=0.8, edgecolor='none')
        
        # Direct labeling instead of legend (Tufte principle)
        if bid_prices and ask_prices:
            # Add subtle text labels
            mid_bid_idx = len(bid_prices) // 2
            mid_ask_idx = len(ask_prices) // 2
            
            if bid_shares:
                ax.text(bid_prices[mid_bid_idx], max(bid_shares) * 0.9, 'Bids', 
                       ha='center', va='bottom', fontsize=9, color='#404040')
            
            if ask_shares:
                ax.text(ask_prices[mid_ask_idx], max(ask_shares) * 0.9, 'Asks', 
                       ha='center', va='bottom', fontsize=9, color='#808080')
        
        # Minimal tick marks
        all_prices = sorted(bid_prices + ask_prices)
        ax.set_xticks(all_prices)
        ax.set_xticklabels([f'${x:.0f}' for x in all_prices], fontsize=9)
        
        # Reduce number of y-ticks for cleaner look
        max_shares = max((bid_shares + ask_shares) if (bid_shares or ask_shares) else [0])
        if max_shares > 0:
            y_ticks = [0, max_shares // 2, max_shares]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=9)
        
        # Minimal axis labels with no unnecessary capitalization
        ax.set_xlabel('price', fontsize=10, color='#666666')
        ax.set_ylabel('shares', fontsize=10, color='#666666')
        
        # No title - let the data speak for itself (Tufte principle)
        
        # Light tick marks
        ax.tick_params(colors='#CCCCCC', length=3, width=0.5)
        
        # Tight layout to maximize data-ink ratio
        plt.tight_layout()
        plt.show()

    def animate_order_operation(self, operation_name: str, operation_details: str, 
                               before_book: 'OrderBook', after_book: 'OrderBook',
                               order_price: Optional[float] = None, order_shares: Optional[int] = None,
                               is_buy: bool = False, is_market: bool = False) -> None:
        """
        Create an interactive animation showing the order book operation step by step.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        import matplotlib.patches as patches
        
        # Tufte-style configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.linewidth': 0.5,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
            'axes.grid': False
        })
        
        # Create figure with space for buttons
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.15)
        
        # Animation state
        current_frame = [0]  # Use list to allow modification in nested functions
        
        # Define frames based on operation type
        if is_market:
            frames = [
                f"Initial State",
                f"Incoming {operation_name}",
                f"Order Execution",
                f"Final State"
            ]
        else:
            frames = [
                f"Initial State", 
                f"Incoming {operation_name}",
                f"Final State"
            ]
        
        def setup_tufte_style(ax):
            """Apply Tufte styling to the axis"""
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            ax.tick_params(colors='#CCCCCC', length=3, width=0.5)
        
        def draw_order_book(ax, book, frame_idx, operation_name, operation_details):
            """Draw the order book for a specific frame"""
            ax.clear()
            setup_tufte_style(ax)
            
            bid_prices = [d_s.dollars for d_s in book.descending_bids]
            bid_shares = [d_s.shares for d_s in book.descending_bids]
            ask_prices = [d_s.dollars for d_s in book.ascending_asks]
            ask_shares = [d_s.shares for d_s in book.ascending_asks]
            
            # Get all prices for consistent x-axis
            all_before_prices = ([d_s.dollars for d_s in before_book.descending_bids] + 
                               [d_s.dollars for d_s in before_book.ascending_asks])
            all_after_prices = ([d_s.dollars for d_s in after_book.descending_bids] + 
                              [d_s.dollars for d_s in after_book.ascending_asks])
            all_prices = sorted(set(all_before_prices + all_after_prices))
            
            # Draw existing order book
            if bid_prices:
                ax.bar(bid_prices, bid_shares, color='#404040', alpha=0.7, 
                      width=0.8, edgecolor='none', label='Bids')
            
            if ask_prices:
                ax.bar(ask_prices, ask_shares, color='#808080', alpha=0.7, 
                      width=0.8, edgecolor='none', label='Asks')
            
            # Frame-specific overlays
            if frame_idx == 1 and order_price is not None and order_shares is not None:
                # Show incoming order
                if is_market:
                    # Market order: show consumption overlay
                    if is_buy:
                        # Highlight asks that will be consumed
                        consumed_prices = []
                        remaining_shares = order_shares
                        for ask in book.ascending_asks:
                            if remaining_shares <= 0:
                                break
                            consumed_prices.append(ask.dollars)
                            remaining_shares -= ask.shares
                        
                        if consumed_prices:
                            ax.bar(consumed_prices, [book.ascending_asks[i].shares for i, p in enumerate([a.dollars for a in book.ascending_asks]) if p in consumed_prices], 
                                  color='#CC4444', alpha=0.4, width=0.8, edgecolor='#CC4444', linewidth=1)
                    else:
                        # Highlight bids that will be consumed
                        consumed_prices = []
                        remaining_shares = order_shares
                        for bid in book.descending_bids:
                            if remaining_shares <= 0:
                                break
                            consumed_prices.append(bid.dollars)
                            remaining_shares -= bid.shares
                        
                        if consumed_prices:
                            ax.bar(consumed_prices, [book.descending_bids[i].shares for i, p in enumerate([b.dollars for b in book.descending_bids]) if p in consumed_prices], 
                                  color='#CC4444', alpha=0.4, width=0.8, edgecolor='#CC4444', linewidth=1)
                else:
                    # Limit order: show new liquidity with diagonal pattern
                    if is_buy:
                        color = '#4444CC'  # Subtle blue for new bids
                    else:
                        color = '#CC6666'  # Light red for new asks
                    
                    # Check if price level already exists
                    existing_shares = 0
                    if is_buy and order_price in bid_prices:
                        idx = bid_prices.index(order_price)
                        existing_shares = bid_shares[idx]
                    elif not is_buy and order_price in ask_prices:
                        idx = ask_prices.index(order_price)
                        existing_shares = ask_shares[idx]
                    
                    # Draw new liquidity with subtle diagonal hatching (Tufte-style)
                    bar = ax.bar([order_price], [order_shares], bottom=[existing_shares], 
                                color=color, alpha=0.3, width=0.8, edgecolor=color, 
                                linewidth=0.5, hatch='///')
            
            elif frame_idx == 2 and is_market:
                # Show consumption in progress
                if is_buy:
                    # Show partially consumed asks
                    consumed_shares = 0
                    remaining_order = order_shares
                    for i, ask in enumerate(book.ascending_asks):
                        if remaining_order <= 0:
                            break
                        take = min(remaining_order, ask.shares)
                        consumed_shares += take
                        remaining_order -= take
                        
                        # Show consumed portion in red
                        ax.bar([ask.dollars], [take], color='#CC4444', alpha=0.6, 
                              width=0.8, edgecolor='none')
                        
                        # Show remaining portion in normal color
                        if ask.shares > take:
                            ax.bar([ask.dollars], [ask.shares - take], bottom=[take],
                                  color='#808080', alpha=0.7, width=0.8, edgecolor='none')
                else:
                    # Show partially consumed bids
                    consumed_shares = 0
                    remaining_order = order_shares
                    for i, bid in enumerate(book.descending_bids):
                        if remaining_order <= 0:
                            break
                        take = min(remaining_order, bid.shares)
                        consumed_shares += take
                        remaining_order -= take
                        
                        # Show consumed portion in red
                        ax.bar([bid.dollars], [take], color='#CC4444', alpha=0.6, 
                              width=0.8, edgecolor='none')
                        
                        # Show remaining portion in normal color
                        if bid.shares > take:
                            ax.bar([bid.dollars], [bid.shares - take], bottom=[take],
                                  color='#404040', alpha=0.7, width=0.8, edgecolor='none')
            
            # Direct labeling (Tufte principle)
            if bid_prices and ask_prices:
                max_height = max((bid_shares + ask_shares) if (bid_shares or ask_shares) else [0])
                if max_height > 0:
                    mid_bid_idx = len(bid_prices) // 2
                    mid_ask_idx = len(ask_prices) // 2
                    
                    if bid_shares:
                        ax.text(bid_prices[mid_bid_idx], max_height * 0.85, 'Bids', 
                               ha='center', va='bottom', fontsize=9, color='#404040')
                    
                    if ask_shares:
                        ax.text(ask_prices[mid_ask_idx], max_height * 0.85, 'Asks', 
                               ha='center', va='bottom', fontsize=9, color='#808080')
            
            # Set consistent axis
            if all_prices:
                ax.set_xticks(all_prices)
                ax.set_xticklabels([f'${x:.0f}' for x in all_prices], fontsize=9)
            
            # Y-axis
            all_shares = (bid_shares + ask_shares) if (bid_shares or ask_shares) else [0]
            if order_shares and frame_idx == 1 and not is_market:
                all_shares.append(order_shares)
            
            max_shares = max(all_shares) if all_shares else 0
            if max_shares > 0:
                y_ticks = [0, max_shares // 2, max_shares]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=9)
            
            # Labels
            ax.set_xlabel('price', fontsize=10, color='#666666')
            ax.set_ylabel('shares', fontsize=10, color='#666666')
            
            # Title with operation info and frame
            title = f"{operation_name} | {operation_details}"
            frame_info = f"Step {frame_idx + 1}/{len(frames)}: {frames[frame_idx]}"
            ax.text(0.02, 0.98, title, transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top', color='#333333', weight='bold')
            ax.text(0.02, 0.92, frame_info, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color='#666666')
        
        def update_plot():
            """Update the plot for the current frame"""
            frame_idx = current_frame[0]
            if frame_idx == 0 or (frame_idx == len(frames) - 1):
                # Show initial or final state
                book_to_show = before_book if frame_idx == 0 else after_book
            else:
                # Show intermediate states with overlays
                book_to_show = before_book
            
            draw_order_book(ax, book_to_show, frame_idx, operation_name, operation_details)
            plt.draw()
        
        def next_frame(event):
            if current_frame[0] < len(frames) - 1:
                current_frame[0] += 1
                update_plot()
        
        def prev_frame(event):
            if current_frame[0] > 0:
                current_frame[0] -= 1
                update_plot()
        
        # Create navigation buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.05])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        
        btn_prev.on_clicked(prev_frame)
        btn_next.on_clicked(next_frame)
        
        # Initial plot
        update_plot()
        
        # Set window title
        fig.canvas.manager.set_window_title(f"Order Book Animation: {operation_name}")
        
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
    
    # Initial order book display (static)
    print("Initial Order Book")
    ob0.display_order_book()

    print("Sell Limit Order of (107, 40)")
    print()
    d_s1, ob1 = ob0.sell_limit_order(107, 40)
    proceeds1: float = d_s1.dollars
    shares_sold1: int = d_s1.shares
    print(f"Sales Proceeds = {proceeds1:.2f}, Shares Sold = {shares_sold1:d}")
    ob1.pretty_print_order_book()
    
    # Animated visualization of the sell limit order
    ob0.animate_order_operation(
        operation_name="Sell Limit Order",
        operation_details=f"Price: $107, Shares: 40, Proceeds: ${proceeds1:.2f}, Sold: {shares_sold1}",
        before_book=ob0,
        after_book=ob1,
        order_price=107,
        order_shares=40,
        is_buy=False,
        is_market=False
    )

    print("Sell Market Order of 120")
    print()
    d_s2, ob2 = ob1.sell_market_order(120)
    proceeds2: float = d_s2.dollars
    shares_sold2: int = d_s2.shares
    print(f"Sales Proceeds = {proceeds2:.2f}, Shares Sold = {shares_sold2:d}")
    ob2.pretty_print_order_book()
    
    # Animated visualization of the sell market order
    ob1.animate_order_operation(
        operation_name="Sell Market Order",
        operation_details=f"Shares: 120, Proceeds: ${proceeds2:.2f}, Sold: {shares_sold2}",
        before_book=ob1,
        after_book=ob2,
        order_price=None,
        order_shares=120,
        is_buy=False,
        is_market=True
    )

    print("Buy Limit Order of (100, 80)")
    print()
    d_s3, ob3 = ob2.buy_limit_order(100, 80)
    bill3: float = d_s3.dollars
    shares_bought3: int = d_s3.shares
    print(f"Purchase Bill = {bill3:.2f}, Shares Bought = {shares_bought3:d}")
    ob3.pretty_print_order_book()
    
    # Animated visualization of the buy limit order
    ob2.animate_order_operation(
        operation_name="Buy Limit Order",
        operation_details=f"Price: $100, Shares: 80, Bill: ${bill3:.2f}, Bought: {shares_bought3}",
        before_book=ob2,
        after_book=ob3,
        order_price=100,
        order_shares=80,
        is_buy=True,
        is_market=False
    )

    print("Sell Limit Order of (104, 60)")
    print()
    d_s4, ob4 = ob3.sell_limit_order(104, 60)
    proceeds4: float = d_s4.dollars
    shares_sold4: int = d_s4.shares
    print(f"Sales Proceeds = {proceeds4:.2f}, Shares Sold = {shares_sold4:d}")
    ob4.pretty_print_order_book()
    
    # Animated visualization of the second sell limit order
    ob3.animate_order_operation(
        operation_name="Sell Limit Order",
        operation_details=f"Price: $104, Shares: 60, Proceeds: ${proceeds4:.2f}, Sold: {shares_sold4}",
        before_book=ob3,
        after_book=ob4,
        order_price=104,
        order_shares=60,
        is_buy=False,
        is_market=False
    )

    print("Buy Market Order of 150")
    print()
    d_s5, ob5 = ob4.buy_market_order(150)
    bill5: float = d_s5.dollars
    shares_bought5: int = d_s5.shares
    print(f"Purchase Bill = {bill5:.2f}, Shares Bought = {shares_bought5:d}")
    ob5.pretty_print_order_book()
    
    # Animated visualization of the buy market order
    ob4.animate_order_operation(
        operation_name="Buy Market Order",
        operation_details=f"Shares: 150, Bill: ${bill5:.2f}, Bought: {shares_bought5}",
        before_book=ob4,
        after_book=ob5,
        order_price=None,
        order_shares=150,
        is_buy=True,
        is_market=True
    )