import subprocess

def run_docker_compose(compose_file_path):
    try:
        process = subprocess.Popen(
            ["docker-compose", "-f", compose_file_path, "up", "--build"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Stream logs in real-time
        for line in process.stdout:
            print(line, end="")
        
        process.wait()  # Wait for the process to complete
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Provide the full path or relative path to the docker-compose.yml file
    compose_file = "ProductUI/docker-compose.yaml"  # Adjust this path
    run_docker_compose(compose_file)
