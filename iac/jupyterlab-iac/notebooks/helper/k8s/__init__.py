import subprocess
import re


# TODO: Make another function called get_most_recent_pod, make sure that it is the one created latest, and also return False if the pos is still in `Pending`
def get_one_running_pod(prefix: str = '') -> str:
    """
    Retrieve the name of the first running pod in the Kubernetes cluster that starts with a given prefix.

    Args:
        prefix (str): The prefix to match the pod name. Only pods with names starting with this prefix will be considered.
                      Defaults to an empty string, meaning all pods are considered.

    Returns:
        str: The name of the first pod in the "Running" state that matches the given prefix.
             If no matching pod is found, returns None.

    Example:
        >>> get_one_running_pod('my-app-')
        'my-app-1234'

        >>> get_one_running_pod()
        'nginx-pod-1'
    """
    try:
        # Execute the `kubectl` command and capture the output
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '--no-headers'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if the command failed
        if result.returncode != 0:
            print(f"Error: {result.stderr.strip()}")
            return None

        # Parse the output
        for line in result.stdout.splitlines():
            if not line.startswith(prefix):
                continue
            fields = re.split(r'\s+', line)
            if len(fields) >= 3 and fields[2] == 'Running':
                return fields[0]
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def get_jupyter_token_from_pod(pod_name: str) -> str:
    """
    Retrieve the Jupyter token from the logs of a specified pod.

    Args:
        pod_name (str): The name of the pod from which to extract the Jupyter token.

    Returns:
        str: The Jupyter token found in the pod's logs. If no token is found, returns None.

    Example:
        >>> get_jupyter_token_from_pod('jupyter-pod-1')
        '123abc456def789ghijk'
    """
    try:
        # Execute the `kubectl logs` command for the given pod
        result = subprocess.run(
            ['kubectl', 'logs', pod_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if the command failed
        if result.returncode != 0:
            print(f"Error fetching logs from pod {pod_name}: {result.stderr.strip()}")
            return None

        # Parse the logs to find the token
        token = None
        for line in result.stdout.splitlines():
            match = re.match(r'.*lab\?token=([a-f0-9]+)', line)
            if match:
                token = match.group(1)
                break
        return token

    except Exception as e:
        print(f"An error occurred: {e}")
        return None