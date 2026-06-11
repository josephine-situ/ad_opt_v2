from utils.campaign_config import load_config_dict


def construct_campaign_name_for_args(course: str, match_type: str, region: str) -> str:
    """Construct campaign name based on course, match type and region."""
    cfg = load_config_dict(course)
    return f"{cfg['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"


def construct_ad_group_name_for_args(course: str, match_type: str, region: str) -> str:
    """Construct ad group name based on course, match type and region."""
    cfg = load_config_dict(course)
    return f"Ad Group - {cfg['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"


def construct_budget_name_for_args(course: str, match_type: str, region: str) -> str:
    """Construct budget name based on course, match type and region."""
    cfg = load_config_dict(course)
    return f"Budget - {cfg['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"

def get_match_types_for_label(label: str) -> list[str]:
    """Return list of match types corresponding to a given label."""
    return [match_type.strip().upper() for match_type in label.split(';')]