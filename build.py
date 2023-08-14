import argparse
import requests
import sys
import os
import string
import json
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
import subprocess

REPO_NAME = "scikit-learn"
REPO_OWNER = "scikit-learn"
TRACKER_REPO = "thomasjpfan/sklearn-pr-tracker"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", default="")

parser = argparse.ArgumentParser()
parser.add_argument("--save_file", help="Set to file to save results locally")
parser.add_argument("--load_file", help="Set to file to load results locally")
args = parser.parse_args()

query_template = string.Template(
    """
query {
  repository(name:"$REPO_NAME", owner:"$REPO_OWNER") {
    pullRequests(
      first: 100,
      after: $CURSOR,
      states: OPEN
    ) {
        nodes {
        title
        updatedAt
        additions
        deletions
        number
        totalCommentsCount
        reviews(last: 1, states:APPROVED) {
          totalCount
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""
)


if args.load_file:
    print("Loading data locally")
    with open(args.load_file, "r") as f:
        results = json.load(f)

else:
    print("Getting data from GitHub")
    url = "https://api.github.com/graphql"

    if not GITHUB_TOKEN:
        print("A GITHUB_TOKEN must be defined to query the GraphQL")
        sys.exit(1)
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

    has_next_page = True
    cursor = "null"
    results = []
    while has_next_page:
        query = query_template.substitute(
            CURSOR=cursor, REPO_NAME=REPO_NAME, REPO_OWNER=REPO_OWNER
        )
        r = requests.post(url=url, json={"query": query}, headers=headers)
        r_json = r.json()

        page_info = r_json["data"]["repository"]["pullRequests"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        if has_next_page:
            cursor = f'"{page_info["endCursor"]}"'
        results.extend(r_json["data"]["repository"]["pullRequests"]["nodes"])

        print(f"Current number of results: {len(results)}")

    if args.save_file:
        with open(args.save_file, "w") as f:
            json.dump(results, f)


print("Processing results")
results_processed = [
    {
        "pr": result["number"],
        "title": f'<a href="https://github.com/scikit-learn/scikit-learn/pull/{result["number"]}">{result["title"]}</a>',
        "additions": result["additions"],
        "deletions": result["deletions"],
        "comments": result["totalCommentsCount"],
        "approvals": result["reviews"]["totalCount"],
        "updated": result["updatedAt"],
    }
    for result in results
]

results_df = pd.DataFrame.from_records(results_processed)
results_df["updated"] = pd.to_datetime(results_df["updated"])

buffer = StringIO()
results_df.to_csv(buffer, index=False)
result_str = buffer.getvalue()

print("Creating index.py page")
with open("index.py.template", "r") as f:
    template = string.Template(f.read())

utc_now = datetime.now(timezone.utc)

target_repo = f"{REPO_OWNER}/{REPO_NAME}"
output = template.substitute(
    CSV_CONTENT=result_str,
    TRACKER_REPO=TRACKER_REPO,
    TARGET_REPO=target_repo,
    DATE=utc_now.strftime("%B %d, %Y"),
)

print("Writing to index.py")
with open("index.py", "w") as f:
    f.write(output)


subprocess.run(
    [
        sys.executable,
        "-m",
        "panel",
        "convert",
        "index.py",
        "--to",
        "pyodide-worker",
        "--out",
        "build",
        "--requirements",
        "requirements.txt",
    ]
)
