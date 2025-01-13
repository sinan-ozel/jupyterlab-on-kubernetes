from typing import List
from botocore import client as EC2


def get_vpcs_ids(ec2_client: EC2, tags: dict) -> List[str]:
    """
    Retrieves the IDs of VPCs that match the specified tags.

    Args:
        ec2_client: An EC2 client object from the AWS SDK (boto3),
                    used to interact with the AWS EC2 service.
        tags (dict): A dictionary containing key-value pairs to match
                     against the tags of VPCs. For example, `{'purpose': 'dev'}`.

    Returns:
        List[str]: A list of VPC IDs that have a tag matching the specified key-value pair.

    Example:
        ec2_client = boto3.client('ec2')
        tags = {'purpose': 'production'}
        vpc_ids = get_vpcs_ids(ec2_client, tags)
        print(vpc_ids)  # Output: ['vpc-12345', 'vpc-67890']

    Notes:
        - The function specifically checks for a tag with the key `'purpose'`
          and matches its value with the one provided in the `tags` dictionary.
        - If the `Tags` key is missing or does not contain the specified key-value pair,
          the VPC is excluded from the result.
        - Ensure the EC2 client has the necessary permissions to call `describe_vpcs`.

    Raises:
        KeyError: If the `tags` dictionary does not contain the key `'purpose'`.
    """
    vpcs_response = ec2_client.describe_vpcs()
    vpc_ids = []
    for vpc in vpcs_response['Vpcs']:
        for tag in vpc.get('Tags', []):
            if tag['Key'] == 'purpose' and tag['Value'] == tags['purpose']:
                vpc_ids.append(vpc['VpcId'])
    return vpc_ids


def get_internet_gateway_ids_attached_to_vpc(ec2_client: EC2, vpc_id: str) -> List[str]:
    """
    Retrieve the IDs of internet gateways attached to a specific VPC.

    Args:
        ec2_client (EC2): An EC2 client instance from Boto3, used to interact with the AWS EC2 service.
        vpc_id (str): The ID of the VPC to find attached internet gateways.

    Returns:
        List[str]: A list of internet gateway IDs attached to the specified VPC. Returns an empty list
                   if no internet gateways are attached.

    Example:
        >>> import boto3
        >>> ec2_client = boto3.client('ec2')
        >>> vpc_id = "vpc-0abcd1234efgh5678"
        >>> get_internet_gateway_ids_attached_to_vpc(ec2_client, vpc_id)
        ['igw-0abcd1234efgh5678']
    """
    response = ec2_client.describe_internet_gateways()
    ids = []
    for ig in response['InternetGateways']:
        for attachment in ig.get('Attachments', []):
            if attachment.get('VpcId', '') == vpc_id:
                ids.append(ig['InternetGatewayId'])
    return ids


def get_route_table_ids_for_vpc(ec2_client: EC2, vpc_id: str):
    """
    Retrieve the IDs of route tables associated with a specific VPC.

    Args:
        ec2_client (EC2): An EC2 client instance from Boto3, used to interact with the AWS EC2 service.
        vpc_id (str): The ID of the VPC for which to retrieve associated route tables.

    Returns:
        List[str]: A list of route table IDs associated with the specified VPC. Returns an empty list
                   if no route tables are associated with the VPC.

    Example:
        >>> import boto3
        >>> ec2_client = boto3.client('ec2')
        >>> vpc_id = "vpc-0abcd1234efgh5678"
        >>> get_route_table_ids_for_vpc(ec2_client, vpc_id)
        ['rtb-0abcd1234efgh5678']
    """
    response = ec2_client.describe_route_tables()
    rt_ids = []
    for route_table in response['RouteTables']:
        if route_table['VpcId'] == vpc_id:
            rt_ids.append(route_table['RouteTableId'])
    return rt_ids


def route_to_gateway_exists(ec2_client: EC2, route_table_id: str, igw_id: str) -> bool:
    """
    Check if a route to a specific internet gateway exists in the given route table.

    Args:
        ec2_client (EC2): An EC2 client instance from Boto3, used to interact with the AWS EC2 service.
        route_table_id (str): The ID of the route table to check.
        igw_id (str): The ID of the internet gateway to check for in the route table.

    Returns:
        bool: True if a route to the specified internet gateway exists in the route table,
              False otherwise.

    Example:
        >>> import boto3
        >>> ec2_client = boto3.client('ec2')
        >>> route_table_id = "rtb-0abcd1234efgh5678"
        >>> igw_id = "igw-0abcd1234efgh5678"
        >>> route_to_gateway_exists(ec2_client, route_table_id, igw_id)
        True
    """
    response = ec2_client.describe_route_tables(RouteTableIds=[route_table_id])
    for route_table in response['RouteTables']:
        for route in route_table['Routes']:
            if route['GatewayId'] == igw_id:
                return True
    return False


def get_subnet_ids_in_vpc(ec2_client: EC2, vpc_id: str) -> List[str]:
    """
    Retrieve the IDs of subnets associated with a specific VPC.

    Args:
        ec2_client (EC2): An EC2 client instance from Boto3, used to interact with the AWS EC2 service.
        vpc_id (str): The ID of the VPC for which to retrieve associated subnets.

    Returns:
        List[str]: A list of subnet IDs associated with the specified VPC. Returns an empty list
                   if no subnets are associated with the VPC.

    Example:
        >>> import boto3
        >>> ec2_client = boto3.client('ec2')
        >>> vpc_id = "vpc-0abcd1234efgh5678"
        >>> get_subnet_ids_in_vpc(ec2_client, vpc_id)
        ['subnet-0abcd1234efgh5678', 'subnet-1abcd1234efgh5678']
    """
    subnets_response = ec2_client.describe_subnets()
    subnet_ids = []
    for subnet in subnets_response['Subnets']:
        if subnet['VpcId'] == vpc_id:
            subnet_ids.append(subnet['SubnetId'])
    return subnet_ids


def get_security_group_ids(ec2_client: EC2, tags: dict) -> List[str]:
    """
    Retrieve the IDs of security groups that match a specified tag.

    Args:
        ec2_client (EC2): An EC2 client instance from Boto3, used to interact with the AWS EC2 service.
        tags (dict): A dictionary containing key-value pairs of tags to filter the security groups.
                     The function looks specifically for a 'purpose' tag with the value matching
                     `tags['purpose']`.

    Returns:
        List[str]: A list of security group IDs that have a 'purpose' tag matching the value
                   specified in `tags['purpose']`. Returns an empty list if no matching security
                   groups are found.

    Example:
        >>> import boto3
        >>> ec2_client = boto3.client('ec2')
        >>> tags = {'purpose': 'web-server'}
        >>> get_security_group_ids(ec2_client, tags)
        ['sg-0abcd1234efgh5678', 'sg-1abcd1234efgh5678']
    """
    response = ec2_client.describe_security_groups()
    ids = []
    for sg in response['SecurityGroups']:
        for tag in sg.get('Tags', []):
            if tag['Key'] == 'purpose' and tag['Value'] == tags['purpose']:
                ids.append(sg['GroupId'])
    return ids