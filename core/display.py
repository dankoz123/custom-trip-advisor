from rich.console import Console
from rich.table import Table, box
from rich.panel import Panel
from core.models import Itinerary
from crewai.types.usage_metrics import UsageMetrics

# Claude Haiku 4.5 pricing (USD per 1M tokens)
_PRICE_INPUT        = 0.80
_PRICE_CACHED_INPUT = 0.08
_PRICE_OUTPUT       = 4.00

console = Console()


def show_itinerary(itinerary: Itinerary) -> None:
    header = f"{itinerary.destination}  ·  {itinerary.start_date} → {itinerary.end_date}  ·  {itinerary.total_days} days"
    console.print()
    console.rule(f"[bold white]{header}[/bold white]", style="dim white")
    console.print()

    for day in itinerary.days:
        day_title = f"Day {day.day_number}  ·  {day.date}  ·  {day.title}"
        console.rule(f"[bold cyan]{day_title}[/bold cyan]", style="cyan")
        console.print(f"[italic]{day.description}[/italic]")
        console.print()

        table = Table(
            box=box.HEAVY_HEAD,
            show_header=True,
            header_style="bold white",
            border_style="dim white",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Time",   style="bold yellow", width=7,  no_wrap=True)
        table.add_column("Place",  style="white",       min_width=30)
        table.add_column("Dur",    style="cyan",        width=9,  no_wrap=True)
        table.add_column("Source", style="blue",        min_width=40)

        for attr in day.attractions:
            table.add_row(attr.time, attr.place, f"{attr.duration_min} min", attr.source)

        console.print(table)
        console.print()


def show_usage(usage: UsageMetrics, model: str) -> None:
    uncached_input = usage.prompt_tokens - usage.cached_prompt_tokens
    cost_input     = (uncached_input             / 1_000_000) * _PRICE_INPUT
    cost_cached    = (usage.cached_prompt_tokens / 1_000_000) * _PRICE_CACHED_INPUT
    cost_output    = (usage.completion_tokens    / 1_000_000) * _PRICE_OUTPUT
    total_cost     = cost_input + cost_cached + cost_output

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="dim white")
    table.add_column(style="bold white", justify="right")

    table.add_row("Model",                   f"[dim]{model}[/dim]")
    table.add_row("─" * 22,                  "─" * 12)
    table.add_row("Prompt tokens",           f"{usage.prompt_tokens:,}")
    table.add_row("  of which cached",       f"{usage.cached_prompt_tokens:,}")
    table.add_row("Completion tokens",       f"{usage.completion_tokens:,}")
    table.add_row("Total tokens",            f"{usage.total_tokens:,}")
    table.add_row("API requests",            f"{usage.successful_requests:,}")
    table.add_row("─" * 22,                  "─" * 12)
    table.add_row("Input cost",              f"${cost_input:.6f}")
    table.add_row("Cached input cost",       f"${cost_cached:.6f}")
    table.add_row("Output cost",             f"${cost_output:.6f}")
    table.add_row("[bold]Total cost[/bold]", f"[bold green]${total_cost:.6f}[/bold green]")

    console.print(Panel(table, title="[bold]Usage & Cost[/bold]", border_style="dim white"))
