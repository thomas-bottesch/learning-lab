import argparse, datetime, json, subprocess, tempfile, requests, yaml


def search_specs(registry: str) -> list[dict]:
    query = '{ GlobalSearch(query: "-spec") { Images { RepoName Tag } } }'
    r = requests.post(
        f"http://{registry}/v2/_zot/ext/search",
        json={"query": query},
        timeout=30,
    )
    r.raise_for_status()
    images = r.json()["data"]["GlobalSearch"]["Images"]
    images = [img for img in images if img["RepoName"].endswith("-spec")]

    from packaging.version import Version, InvalidVersion

    best: dict[str, dict] = {}
    for img in images:
        repo = img["RepoName"]
        try:
            v = Version(img["Tag"])
        except InvalidVersion:
            continue
        if repo not in best or v > Version(best[repo]["Tag"]):
            best[repo] = img
    return list(best.values())


def pull_manifest(registry: str, repo: str, tag: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "oras",
                "pull",
                "--plain-http",
                f"{registry}/{repo}:{tag}",
                "--output",
                tmpdir,
            ],
            check=True,
            capture_output=True,
        )
        with open(f"{tmpdir}/manifest.yaml") as f:
            return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()

    specs = search_specs(args.registry)
    components = []
    for img in specs:
        manifest = pull_manifest(args.registry, img["RepoName"], img["Tag"])
        name = img["RepoName"].removeprefix("ml-components/").removesuffix("-spec")
        components.append(
            {
                "name": name,
                "latest_stable": img["Tag"],
                "spec_ref": f"{args.registry}/{img['RepoName']}:{img['Tag']}",
                "category": manifest.get("category", ""),
                "tags": manifest.get("tags", []),
                "description": manifest.get("description", ""),
                "typical_downstream": manifest.get("typical_downstream", []),
                "outputs": manifest.get("outputs", []),
            }
        )

    try:
        with open("catalog_patterns.json") as f:
            patterns = json.load(f)
    except FileNotFoundError:
        patterns = {"composition_patterns": []}

    catalog = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "components": components,
        "composition_patterns": patterns.get("composition_patterns", []),
    }
    with open("catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog written: {len(components)} components.")


if __name__ == "__main__":
    main()
