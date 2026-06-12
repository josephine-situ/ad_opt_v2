from google.ads.googleads.v23.enums import AgeRangeTypeEnum

# Map CSV age ranges to Google Ads age range types
AGE_RANGE_MAP = {
    "18 - 24": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24,
    "25 - 34": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34,
    "35 - 44": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_35_44,
    "45 - 54": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_45_54,
    "55 - 64": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_55_64,
    "65+": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_65_UP,
    "Unknown": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_UNDETERMINED,
}
