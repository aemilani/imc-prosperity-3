# IMC Prosperity 3 🐚

This repository contains the algorithmic trading code and manual challenge solutions for our team, `Arbland`, in **IMC Prosperity 3 (2025)**.

Over the 15 days of the competition, we developed a highly diversified algorithmic trading architecture—ranging from standard inventory-managed market making to statistical arbitrage and options volatility trading. Alongside the algorithm, we tackled five manual math and game-theory challenges using numerical optimization and Monte Carlo simulations.

## Architecture & State Management

To keep our bot performant and our logic isolated, we heavily utilized Python `dataclasses` to define the state, constraints, and Greeks of each individual product.

Because the Prosperity environment is stateless between ticks, we serialized our rolling windows, historical prices, and implied volatilities into the `traderData` string using `jsonpickle`. This allowed us to maintain continuous moving averages and polynomial fits across the trading periods without breaking execution time limits.

## Algorithmic Trading Strategies

Instead of relying on a single alpha signal, we deployed bespoke models tailored to the specific micro-structure and behavior of each product.

### Kelp & Rainforest Resin: Mean-Reversion & Inventory Management

For these base assets, we acted as aggressive market makers.

* **Rainforest Resin:** We used a hardcoded fair value ($10,000) and executed a standard pennying/joining strategy. We implemented a `soft_pos_limit` to skew our quotes: if our inventory grew too heavy, we aggressively lowered the ask or raised the bid to induce clearing.
* **Kelp:** We filtered out order book noise by ignoring levels with a volume of less than 15. We then modeled the fair price as a mean-reverting stochastic process, forecasting the next fair value ($E[P_{t+1}]$) using the log return between the current mid-price and the previous time step:

$$E[P_{t+1}] = P_{mid} \cdot e^{-0.27 \cdot \ln(P_{mid}/P_{t-1})}$$

### Squid Ink: Avellaneda-Stoikov Market Making

For `SQUID_INK`, we abandoned threshold-based quoting in favor of a theoretical Avellaneda-Stoikov model. We calculated a dynamic reservation price ($r$) that shifted away from the fair market value ($s$) based on our current inventory ($q$), risk aversion ($\gamma = 0.2$), and fixed volatility ($\sigma = 0.5$):

$$r = s - q \cdot \gamma \cdot \sigma^2$$

We placed quotes symmetrically around this reservation price at a fixed spread. This automatically skewed our bid-ask placement to control inventory risk without requiring hardcoded limits.

### Spread Basket: Statistical Arbitrage

We traded `SPREAD` as a synthetic asset composed of a predefined, cointegrated basket: `PICNIC_BASKET1`, `PICNIC_BASKET2`, `CROISSANTS`, `JAMS`, and `DJEMBES`.

* We assumed a long/short relationship with weights $(1, -1, -2, -1, -1)$.
* We computed the synthetic mid-price of this combined order book and calculated a Z-score using a historical mean ($-202.3$) and standard deviation ($83.9$).
* If the Z-score breached our confidence threshold ($\pm 0.8$), we aggressively bought or sold the spread up to our position limits.

### Volcanic Rock & Vouchers: Volatility Skew & Delta Hedging

This was our most computationally heavy component, effectively acting as an automated relative-value options desk.

* **Pricing Engine:** We implemented a custom Black-Scholes class to calculate theoretical prices and Greeks (Delta, Gamma, Vega). We utilized a bisection search algorithm to back out Implied Volatility (IV) from market prices.
* **Volatility Smile:** We tracked a rolling 30-period window of moneyness ($M$) and implied volatilities ($V$) for options with meaningful Vega. We fit a second-degree polynomial to this data ($IV = aM^2 + bM + c$) to model the volatility smile, using the y-intercept ($c$) as our At-The-Money `base_iv`.
* **Execution & Hedging:** We calculated a rolling Z-score of this `base_iv`. When the IV spread deviated by more than 1 standard deviation, we shorted or longed the two options closest to the money. To isolate this volatility exposure, we continuously calculated the net delta of the entire options portfolio and executed offsetting spot trades in the underlying `VOLCANIC_ROCK` to maintain strict delta neutrality.

### Magnificent Macarons: Signal Trading & Dynamic Edge

Macarons required observing external variables (tariffs, transport fees) to establish implied bids and asks.

* **Sunlight Signal:** We tracked changes in the `sunlightIndex`. A drop of $-0.02$ or more acted as a strong buy signal, prompting the algorithm to immediately cross the spread and take the best available asks.
* **Dynamic Spread:** When no sunlight signal was present, we attempted to capture edge over the implied ask. We used a dynamic adjustment mechanism based on a 5-period rolling average of position sizes. If we were successfully turning over larger positions ($\ge 7$), we widened our required edge by $0.5$. If volume dropped, we tightened the edge back down to remain competitive.

## Manual Challenges

The manual rounds tested our ability to apply game theory, probability, and optimization to static datasets.

* **Round 1 (Arbitrage):** A graph traversal problem. We mapped the tradable items to an integer dictionary and wrote a search algorithm to calculate compound exchange rates across all possible 5-trade sequences. The optimal path yielded: `Sea Shells --> Snowballs --> Silicon Nuggets --> Pizzas --> Snowballs --> Sea Shells`.
* **Round 2 & 4 (Game Theory & Resource Allocation):** A treasure-hunting grid where the payout was diluted by competitors. Instead of assuming a naive uniform distribution of competitors, we built a Monte Carlo simulation. We shaped competitor distributions using polynomial decay sequences to simulate "crowding" on high-value nodes, adding random noise for irrational market behavior. After 10,000 iterations, we isolated the nodes (60, 73, and 79) that consistently yielded over 100,000 in profit regardless of the exact competitor distribution.
* **Round 3 (Optimal Pricing):** We maximized expected profit against a disjoint uniform distribution (values strictly between 160-200 or 250-320). We discretized the price range ($step = 0.001$) and manually constructed the Probability Density Function to compute the expected profit curve and locate the global maximum reserve price.
* **Round 5 (Capital Allocation):** A portfolio optimization problem constrained by a hidden, non-linear fee structure. We used `scipy.optimize.curve_fit` on the provided data to back out the cost formula, revealing a purely quadratic relationship: $Fee = 120x^2$ (where $x$ is the percentage of capital deployed). We then simulated PnL parabolas for different expected returns, mapping the peak profitability for 9 products to allocate exactly 87% of our capital for an expected PnL of 210,500.

## Final Thoughts

Prosperity 3 was an incredible test of both quantitative research and software engineering under strict constraints. Balancing standard market making with complex options pricing and statistical simulations in a stateless environment was highly challenging but immensely rewarding. We look forward to the next iteration of the competition!
