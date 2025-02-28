#!/usr/bin/env python3
import os
import runpod
import subprocess
import requests
import tempfile
import shutil

def handler(job):
    """Build a Docker image and push it to a registry"""
    job_input = job["input"]
    
    # Extract parameters
    context_url = job_input.get("context_url")
    dockerfile = job_input.get("dockerfile", "Dockerfile")
    repository = job_input.get("repository")
    tag = job_input.get("tag", "latest")
    
    # Docker credentials
    docker_username = job_input.get("docker_username")
    docker_password = job_input.get("docker_password")
    
    if not context_url or not repository:
        return {"error": "Missing required parameters"}
    
    # Create working directory
    work_dir = tempfile.mkdtemp()
    
    try:
        # Download build context
        context_tar = os.path.join(work_dir, "context.tar.gz")
        r = requests.get(context_url)
        with open(context_tar, 'wb') as f:
            f.write(r.content)
        
        # Extract context
        subprocess.run(["tar", "-xzf", context_tar, "-C", work_dir], check=True)
        
        # Login to Docker Hub
        if docker_username and docker_password:
            subprocess.run(
                ["docker", "login", "-u", docker_username, "--password-stdin"],
                input=docker_password.encode(),
                check=True
            )
        
        # Build the image
        full_tag = f"{repository}:{tag}"
        subprocess.run(
            ["docker", "build", "-f", os.path.join(work_dir, dockerfile), "-t", full_tag, work_dir],
            check=True
        )
        
        # Push the image
        subprocess.run(["docker", "push", full_tag], check=True)
        
        return {
            "success": True,
            "image": full_tag
        }
        
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Clean up
        shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler}) 