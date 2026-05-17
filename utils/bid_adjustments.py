from google.ads.googleads.v23.enums import AgeRangeTypeEnum, DeviceEnum

AGE_RANGE_MAP = {
    "18 - 24": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24,
    "25 - 34": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34,
    "35 - 44": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_35_44,
    "45 - 54": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_45_54,
    "55 - 64": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_55_64,
    "65+": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_65_UP,
    "Unknown": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_UNDETERMINED,
}

AGE_ENUM_TO_RANGE = {value: key for key, value in AGE_RANGE_MAP.items()}

DEVICE_MAP = {
    "Mobile phones": DeviceEnum.Device.MOBILE,
    "Tablets": DeviceEnum.Device.TABLET,
    "Computers": DeviceEnum.Device.DESKTOP,
    "Connected TV": DeviceEnum.Device.CONNECTED_TV,
    "Other": DeviceEnum.Device.OTHER,
}

DEVICE_ENUM_TO_NAME = {value: key for key, value in DEVICE_MAP.items()}
