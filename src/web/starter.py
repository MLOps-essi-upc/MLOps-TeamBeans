import subprocess

def get_docker_status():
    docker_status = input("Is Docker running locally or externally? (local/external): ")
    return docker_status.lower()

def get_public_ip():
    public_ip = input("Enter the public IP: ")
    return public_ip

def save_public_ip(public_ip):
    # Save the public IP to a file or database
    # You can customize this function based on your requirements
    # For example, you can save it to a file using the following code:
    with open("public_ip.txt", "w") as file:
        file.write(public_ip)

def run_app(public_ip):
    # Run app.py passing the public IP as an argument
    # MD change: to python3 
    subprocess.run(["python3", "app.py", public_ip])

def main():
    # docker_status = get_docker_status()
    # if docker_status == "external":
    #     public_ip = get_public_ip()
    #     save_public_ip(public_ip)
    # else:
    public_ip = "localhost"
    run_app(public_ip)

if __name__ == "__main__":
    main()
