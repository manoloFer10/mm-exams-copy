import io
import requests
import pandas as pd


def fetch_spreadsheet(spreadsheet_id: str, gid: int | None = None) -> pd.DataFrame:
    request_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"

    if gid is not None:
        request_url += f"&gid={gid}"

    response = requests.get(request_url)

    return pd.read_csv(io.StringIO(response.text))


def get_related_contributors(contributors: pd.DataFrame, name: str, discord_handle: str) -> pd.DataFrame:
    return contributors[
        (contributors["Name"] == name) |
        (contributors["Discord Id"].isin([discord_handle, f"@{discord_handle}"]))
    ]


def main():
    contributors = fetch_spreadsheet("1XrkpSsJdD0nGg3mCAVhDYAxYdCDgBUdJ5saxsjpfIJo")
    exams = fetch_spreadsheet("1f4nkmFyTaYu0-iBeRQ1D-KTD3JoyC-FI7V9G6hTdn5o", 1644933322)

    aggregated_metadata = []

    for _, exam in exams.iterrows():
        dataset_link = exam.get("HF Dataset Link")

        if pd.isna(dataset_link):
            continue

        contributor_name = exam.get("Contributor")
        contributor_discord_handle = exam.get("Discord Handle").lstrip("@")

        related_contributors = get_related_contributors(contributors, contributor_name, contributor_discord_handle)

        if len(related_contributors) == 0:
            raise ValueError(f"No contributors found for {dataset_link}")

        if len(related_contributors) > 1:
            raise ValueError(f"Multiple contributors found for {dataset_link}")

        contributor_country = related_contributors.iloc[0].get("Country")

        if pd.isna(contributor_country):
            raise ValueError(f"No country has been set for {dataset_link}")

        aggregated_metadata.append((dataset_link, contributor_name, contributor_country))

    aggregated_data = pd.DataFrame(aggregated_metadata, columns=["dataset_link", "contributor_name", "contributor_country"])

    aggregated_data.to_json("dataset/datasets_metadata.json", orient="records", indent=4)


if __name__ == "__main__":
    main()
