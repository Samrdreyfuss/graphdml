"""End-to-end Neo4j demo: seed a dataset, estimate effects from Neo4j, write results back.

    docker compose up -d
    python -m graphdml.demo --password graphdml-demo [--dataset flu_town]

Then open Neo4j Browser (http://localhost:7474) and run the printed queries.
"""

from __future__ import annotations

import argparse
import warnings

from graphdml import GraphDML, datasets
from graphdml.exceptions import GraphDMLWarning
from graphdml.io.neo4j import Neo4jGraphSource, Neo4jResultWriter, connect, write_dataset

DATASETS = {
    # name: (factory, node label, relationship type)
    "flu_town": ("make_flu_town", "Resident", "NEIGHBOR_OF"),
    "referral_app": ("make_referral_app", "User", "FRIENDS_WITH"),
    "classroom_tutoring": ("make_classroom_tutoring", "Student", "FRIENDS_WITH"),
    "homophily_trap": ("make_homophily_trap", "Person", "FRIENDS_WITH"),
    "lastfm_promo": ("make_lastfm_promo", "Listener", "FOLLOWS_MUTUALLY"),
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--uri", default="bolt://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", required=True)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="flu_town")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)

    factory, label, rel = DATASETS[args.dataset]
    ds = getattr(datasets, factory)()
    run_name = args.run_name or f"{args.dataset}-demo"
    driver = connect(args.uri, (args.user, args.password))
    try:
        print(f"1. Seeding {ds.data.n_nodes:,} :{label} nodes and {ds.data.n_edges:,} "
              f":{rel} relationships ...")
        write_dataset(ds, driver, node_label=label, rel_type=rel, replace=True)

        print("2. Loading the graph back from Neo4j ...")
        data = Neo4jGraphSource(driver).load(
            label, rel, features=list(ds.data.feature_names), treatment=ds.treatment_name,
            outcome=ds.outcome_name, id_property="node_id",
        )

        print("3. Estimating direct and peer effects ...\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GraphDMLWarning)
            model = GraphDML(exposure=ds.exposure, random_state=0).fit(data)
        print(model.summary())
        print(f"\nTrue effects: {ds.truth}\n")

        writer = Neo4jResultWriter(driver)
        writer.delete_run(run_name)
        out = writer.write(model, run_name=run_name, node_label=label, id_property="node_id",
                           description=f"graphdml demo on {args.dataset}")
        print(f"4. Wrote run {run_name!r}: {out['written']:,} focal nodes linked.\n")
        print("Explore in Neo4j Browser:")
        print(f"  // the run and its estimates\n  MATCH (r:GDMLRun {{name: '{run_name}'}}) "
              "RETURN r")
        print(f"  // focal nodes (the effective sample) and their neighborhoods\n"
              f"  MATCH (r:GDMLRun {{name: '{run_name}'}})-[e:ESTIMATED_ON]->(n)"
              f"-[:{rel}]-(m) RETURN n, m LIMIT 300")
        print(f"  // largest outcome residuals\n  MATCH (:GDMLRun {{name: '{run_name}'}})"
              "-[e:ESTIMATED_ON]->(n) RETURN n.node_id, e.res_y ORDER BY abs(e.res_y) DESC "
              "LIMIT 10")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
