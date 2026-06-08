from typing import TypedDict


class CourseConfig(TypedDict, total=False):
    min_date: str
    regions: dict[str, list[str]]
    conversion_actions: list[str]
    purchase_actions: list[str]
    start_dates: list[str]
    campaign_budget: float
    current_campaign_start_date: str
    current_campaign_end_date: str


REGION_CONFIG = {
    "sys_think": {
        "USA": ["United States"],
        "A": [
            "France",
            "Switzerland",
            "Sweden",
            "Canada",
            "New Zealand",
            "Netherlands",
            "United Kingdom",
            "Japan",
            "Spain",
            "Denmark",
            "Australia",
            "Ireland",
            "Germany",
            "Norway",
            "Belgium",
            "Mexico",
            "Italy",
        ],
        "B": [
            "Indonesia",
            "Philippines",
            "Uganda",
            "Morocco",
            "Tanzania",
            "Zimbabwe",
            "Tunisia",
            "Sri Lanka",
            "Liberia",
            "Thailand",
            "Turkiye",
            "Cote d'Ivoire",
            "Peru",
            "Jordan",
            "Argentina",
            "Senegal",
            "Saudi Arabia",
            "Malaysia",
            "Singapore",
            "Colombia",
            "Chile",
            "Armenia",
            "China",
            "Georgia",
            "Hong Kong",
            "Israel",
            "Romania",
            "Bulgaria",
            "Trinidad and Tobago",
            "Serbia",
            "Poland",
            "Lithuania",
            "Greece",
            "Qatar",
            "Bolivia",
            "Portugal",
            "Bahrain",
            "Paraguay",
            "Austria",
            "Hungary",
            "Moldova",
            "United Arab Emirates",
            "Czechia",
            "South Korea",
            "Taiwan",
            "Croatia",
            "Estonia",
            "Iceland",
            "Slovakia",
            "Finland",
            "Luxembourg",
            "Monaco",
        ],
    },
}


COURSE = "sys_think"

COURSE_CONFIG: dict[str, CourseConfig] = {
    "sys_think": {
        "min_date": "2024-06-01",
        "start_dates": [
            "2021-01-25",
            "2021-04-05",
            "2021-10-04",
            "2022-01-31",
            "2022-04-25",
            "2022-10-03",
            "2023-01-30",
            "2023-04-10",
            "2023-10-02",
            "2024-02-05",
            "2024-04-08",
            "2024-10-07",
            "2025-02-10",
            "2025-04-14",
            "2025-10-06",
            "2026-02-02",
            "2026-04-06",
            "2026-06-15",
        ],
        "campaign_budget": 412.7,
        "current_campaign_start_date": "2026-04-28",
        "current_campaign_end_date": "2026-06-15",
        "regions": REGION_CONFIG["sys_think"],
        "conversion_actions": [
            "Purchase",
            "System Thinking - Add to cart",
            "idimension - account create",
            "Add to cart - iDimension",
            "HubSpot - Customers",
        ],
        "purchase_actions": ["Purchase", "HubSpot - Customers"],
    },
}
