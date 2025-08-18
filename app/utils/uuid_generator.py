import datetime


def generate_uuid(uuid_type='id'):
    """
    Generate a human-readable UUID for distributed systems

    Args:
        uuid_type (str): Type of UUID (defaults to 'id')

    Returns:
        str: UUID in format {type}_{date}_{human_readable_utc_time}_{milliseconds}
    """
    # Get current UTC time
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # Format date as YYYYMMDD
    date_str = now_utc.strftime('%Y%m%d')

    # Format time as HHhMMmSSs
    time_str = now_utc.strftime('%Hh%Mm%Ss')

    # Get milliseconds for uniqueness
    milliseconds = now_utc.microsecond // 1000

    # Create UUID with milliseconds for guaranteed uniqueness
    uuid = f"{uuid_type}_{date_str}_{time_str}_{milliseconds:03d}"

    return uuid
