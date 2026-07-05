"""CLI con salida elegante (Rich)."""
from __future__ import annotations

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .config import Settings, setup_logging
from . import pipeline, viz

console = Console()


def _tests_table(t: dict) -> Table:
    tab = Table(title="Pruebas de aleatoriedad / uniformidad", box=box.SIMPLE_HEAVY)
    tab.add_column("Prueba"); tab.add_column("Estadistico"); tab.add_column("p-valor"); tab.add_column("Lectura")
    cw, pw, dfw = t["chi2_white"]; cp, pp, dfp = t["chi2_pb"]
    ksd, ksp = t["ks_sums"]; runs, mur, zr, pr = t["runs_test"]
    H, Hmax, Hr = t["entropy_white"]; p11, p10, br = t["markov"]
    tab.add_row("Chi2 blancas", f"{cw:.2f} (gl {dfw})", f"{pw:.3f}",
                "uniforme" if pw > .05 else "rechaza")
    tab.add_row("Chi2 Powerball", f"{cp:.2f} (gl {dfp})", f"{pp:.3f}",
                "uniforme" if pp > .05 else "rechaza")
    tab.add_row("KS sumas~Normal", f"D={ksd:.3f}", f"{ksp:.3f}",
                "~normal" if ksp > .05 else "rechaza")
    tab.add_row("Runs (Wald-Wolfowitz)", f"z={zr:+.2f}", f"{pr:.3f}",
                "aleatorio" if pr > .05 else "patron")
    tab.add_row("Entropia blancas", f"{H:.3f}/{Hmax:.3f} bits", f"{100*Hr:.2f}%", "casi maxima")
    tab.add_row("Markov (memoria)", f"{p11:.3f} vs {p10:.3f}", "-", "sin memoria")
    return tab


def main():
    setup_logging()
    args = sys.argv[1:]
    s = Settings(path_2016=args[0], path_2010=args[1]) if len(args) >= 2 else Settings()

    console.print(Panel.fit("[bold]SIMULADOR ESTADISTICO AVANZADO DE POWERBALL[/bold]\n"
                            "[yellow]Powerball es aleatorio. Esto NO predice el sorteo "
                            "ni mejora tu probabilidad real de ganar.[/yellow]",
                            border_style="cyan"))
    res = pipeline.run(s)

    console.print(f"\n[bold]Datos:[/bold] {res.validation['draws_total']} sorteos | "
                  f"fuera de rango: {res.validation['out_of_range_rows']} | "
                  f"nulos: {res.validation['nulls']}")
    console.print(res.eras.to_string(index=False))
    console.print(_tests_table(res.tests))

    # Optimizadores
    opt = Table(title="Optimizadores (deben converger)", box=box.SIMPLE_HEAVY)
    opt.add_column("Metodo"); opt.add_column("Combinacion"); opt.add_column("SCS")
    for name, key in [("Exhaustivo (global)", "exhaustive"),
                      ("DEAP (genetico)", "deap_ga"),
                      ("Simulated Annealing", "annealing")]:
        combo, sc = res.optimizers[key]
        opt.add_row(name, " ".join(f"{x:02d}" for x in combo), f"{sc:.2f}")
    console.print(opt)

    if "error" not in res.pymc:
        console.print(f"[bold]PyMC (MCMC):[/bold] {100*res.pymc['pct_credible_contains_uniform']:.0f}% "
                      f"de los intervalos creibles del Powerball contienen 1/26 (uniforme).")

    # Jugada final
    play = Table(title="JUGADA FINAL (Statistical Confidence Score)",
                 box=box.DOUBLE_EDGE)
    play.add_column("#"); play.add_column("Blancas"); play.add_column("PB"); play.add_column("SCS")
    for _, row in res.plays.iterrows():
        play.add_row(str(row["jugada"]), f"[bold green]{row['blancas']}[/bold green]",
                     str(row["powerball"]), f"{row['scs']:.2f}")
    console.print(play)

    html = viz.dashboard(res.analysis, res.model, res.exhaustive[0])
    console.print(f"[dim]Dashboard Plotly guardado en {html}[/dim]")
    console.print(f"[bold red]Probabilidad real de jackpot: 1 entre "
                  f"{s.rules.jackpot_odds:,} (igual para toda combinacion).[/bold red]")


if __name__ == "__main__":
    main()
