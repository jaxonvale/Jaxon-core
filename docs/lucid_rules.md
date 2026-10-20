# Lucid Trading — rules for 25K and 50K accounts

Source: Lucid's official help center (support.lucidtrading.com), read 2026-10-20. Rules change, so
re-check before buying. Prices are not published in the help center; third-party sites disagree
(roughly $90–$330 depending on plan), so confirm at checkout.

## Plans
| plan | path | notes |
|---|---|---|
| **LucidFlex** | evaluation → funded | 40% consistency in eval (50% paid add-on), optional DLL, no consistency once funded, scaling plan |
| **LucidPro** | evaluation → funded | no eval consistency, DLL on 50K, 40% payout consistency when funded, payout buffer |
| **LucidDaily** | evaluation → funded | 50% consistency, intraday or EOD drawdown in eval (funded is always intraday), **news trading is a hard breach** |
| **LucidDirect** | straight to funded | no evaluation, 20% payout consistency |

## Evaluations
| | Flex 25K | Flex 50K | Pro 25K | Pro 50K | Daily 25K | Daily 50K |
|---|---|---|---|---|---|---|
| Profit target | $1,250 | $3,000 | $1,250 | $3,000 | $1,250 | $3,000 |
| Max loss limit (MLL) | $1,000 | $2,000 | $1,000 | $2,000 | $1,000 | $2,000 |
| Drawdown type | EOD trailing | EOD trailing | EOD trailing | EOD trailing | intraday or EOD (your choice) | intraday or EOD |
| MLL locks at | $25,100 once balance passes $26,100 | $50,100 once balance passes $52,100 | same | same | same | same |
| Daily loss limit | optional | optional | none | $1,200 | $600 | $1,200 |
| Consistency | 40% (50% add-on) | 40% (50% add-on) | none | none | 50% | 50% |
| Max size | 2 minis / 20 micros | 4 minis / 40 micros | 2 / 20 | 4 / 40 | 2 / 20 | 4 / 40 |
| Time limit | none | none | none | none | none listed | none listed |
| Fee model | one-time, no monthly rebill, no activation fee | | | | | |

- **Consistency** = largest single-day profit ÷ total profit. Going over doesn't fail you; you keep trading until it's back under.
- **All daily loss limits are soft**: trading pauses until the next session; the account is only lost at the MLL.
- **EOD drawdown**: the MLL is recalculated from the highest *closing* balance each session; the account is breached when the balance reaches the MLL.

## Funded accounts (25K / 50K)
| | Flex | Pro | Direct | Daily |
|---|---|---|---|---|
| Drawdown | EOD, $1,000 / $2,000 | EOD, $1,000 / $2,000 | EOD, $1,000 / $2,000 | **intraday**, $1,000 / $2,000 |
| DLL | optional | optional: fixed $600 / $1,200, or 60% of peak EOD profit | none / $1,200 then 60% scaling | $600 / $1,200 |
| Max size | scales: 25K 1→2 minis; 50K 2→3→4 minis by profit | 2 / 4 minis | 2 / 4 minis | 2 / 4 minis |
| Payout consistency | none | 40% per cycle | 20% per cycle | none |
| Payouts | 5 profitable days ($100 / $150 min), 50% of profit up to $1,000 / $2,000, up to 5 payouts then live | profit goal $250 / $500; buffer $26,100 / $52,100; cap $1,000→$1,500 / $2,000→$2,500 | goal $1,500→$1,250 / $3,000→$2,500; cap $1,000 / $2,000–$2,500 | see Lucid |
| Split | 90/10 | 90/10 | 90/10 | |
| Min payout request | $500 | $500 | $500 | |

## Platform-wide rules
- **Trading hours:** flat by **4:45 PM ET** Mon–Fri (Lucid auto-closes; holding past it doesn't fail you). Trading reopens 6:00 PM ET. Close earlier on early-close holidays.
- **News:** allowed on Flex, Pro and Direct. **Red-folder news on LucidDaily = hard breach.**
- **Automated trading / bots / trade copiers: allowed.** You're responsible for errors.
- **Prohibited:** microscalping (flagged when >50% of profit comes from trades held ≤5 seconds), high-frequency trading, hedging (opposite sides of the same or correlated contracts across accounts), exploiting sim/price errors.
- **Allowed:** scaling in, DCA (martingale discouraged), scalping in good faith, flipping.
- **Inactivity:** an account with no trade producing at least $1 of P&L in 30 calendar days is deleted.
- **Countries:** the US is not restricted.

## How JAXON maps to these rules
| rule | how it's handled |
|---|---|
| Flat by 4:45 PM ET | bot flattens at 3:40 PM CT (4:40 PM ET), no new entries after 3:30 PM CT |
| Max size | contract cap 20 micros (25K) / 40 micros (50K) |
| EOD trailing MLL + lock | simulated exactly (breach checked on marked-to-market equity at every 1-minute bar, the conservative reading) |
| Consistency | Flex 40%, Daily 50%: target rises to best day ÷ pct |
| Daily loss limit | soft at Lucid; the bot's own daily lockout (1% of start: $250 / $500) is tighter, so it's never reached |
| News (LucidDaily) | **not modeled**: the bot has no news calendar, so LucidDaily results are optimistic |
| Bots allowed | yes, the bot can run on Flex/Pro/Direct |
| Microscalping / HFT | not an issue: the bot trades a few times a day and holds minutes to hours |
