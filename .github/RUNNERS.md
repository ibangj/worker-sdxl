# Setting Up Self-Hosted GitHub Actions Runners on AWS

This guide describes how to set up GitHub Actions self-hosted runners on AWS using Terraform. These runners provide:

- **Large disk space**: 100GB of ephemeral storage per runner for Docker builds
- **Auto-scaling**: Runners automatically scale up and down based on workflow demand
- **Cost optimization**: Scale to zero when no workflows are running
- **Isolation**: Each job runs in its own isolated environment

## Why Self-Hosted Runners?

GitHub-hosted runners have a disk space limitation of 14GB, which is insufficient for building and pushing Docker images for the worker-sdxl project. Self-hosted runners on AWS provide the following advantages:

1. **More disk space**: 100GB per runner
2. **Cost control**: Pay only for what you use
3. **Faster builds**: Runners use AWS Fargate with dedicated compute resources
4. **Better security**: Each job runs in an isolated environment
5. **Customization**: Configure runners for specific needs

## Prerequisites

1. **AWS Account**: You need an AWS account with permissions to create resources like Lambda, ECS, IAM, etc.
2. **GitHub App**: You'll need to create a GitHub App for authentication
3. **Terraform**: Installed locally or available via CI/CD
4. **AWS CLI**: For easier management (optional)

## Setup Steps

### 1. Create a GitHub App

1. Go to GitHub Settings → Developer settings → GitHub Apps → New GitHub App
2. Fill in the basic information:
   - **Name**: "AWS GitHub Runners for [YourRepo]"
   - **Homepage URL**: Your repository URL
   - **Webhook URL**: Leave blank (we'll update later)
   - **Webhook secret**: Generate a random string

3. Set the following permissions:
   - **Repository permissions**:
     - **Actions**: Read & write
     - **Metadata**: Read-only
     - **Checks**: Read & write
   - **Organization permissions** (if using org runners):
     - **Self-hosted runners**: Read & write

4. Generate and download a private key
5. Install the app on your repository or organization

### 2. Prepare AWS Credentials

Create an IAM user or role with appropriate permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "iam:*",
        "lambda:*",
        "logs:*",
        "ec2:*",
        "ecs:*",
        "ecr:*",
        "ssm:*",
        "cloudwatch:*",
        "s3:*"
      ],
      "Resource": "*"
    }
  ]
}
```

Note: In production, you should limit these permissions to specific resources.

### 3. Set Up Terraform Configuration

1. Navigate to the `terraform/runners` directory
2. Create a `terraform.tfvars` file (or copy from the example):

```bash
# Windows
Copy-Item terraform.tfvars.example terraform.tfvars

# Linux/macOS
cp terraform.tfvars.example terraform.tfvars
```

3. Edit the `terraform.tfvars` file with your specific values:
   - AWS region and VPC/subnet IDs
   - GitHub App details (ID, encoded key, etc.)
   - Repository or organization information

### 4. Deploy Infrastructure

#### Using the Provided Scripts

**On Windows:**
```powershell
.\setup.ps1
```

**On Linux/macOS:**
```bash
./setup.sh
```

#### Manual Deployment
```bash
terraform init
terraform plan
terraform apply
```

### 5. Configure GitHub App Webhook

After deployment, Terraform will output a webhook URL. Update your GitHub App with this URL:

1. Go to GitHub → Settings → Developer settings → GitHub Apps
2. Select your GitHub App
3. Update the Webhook URL with the URL from the Terraform output
4. Save changes

## Using Self-Hosted Runners in Workflows

To use the self-hosted runners in your workflows, specify `runs-on: self-hosted` in your GitHub workflow YAML files:

```yaml
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t myapp:latest .
```

## Additional Configuration

### Runner Labels

The runners have the following labels automatically:
- `self-hosted`
- `large-disk`
- `fargate`

You can specify these labels in your workflow:

```yaml
jobs:
  build:
    runs-on: [self-hosted, large-disk]
    # ...
```

### Resource Allocation

The runners are configured with:
- 4 vCPUs
- 8GB RAM
- 100GB ephemeral storage

### Cost Management

To manage costs:
- Runners automatically scale to zero when not in use
- The cleanup script (`cleanup.sh` or `cleanup.ps1`) can be used to remove all resources when not needed

## Troubleshooting

If you encounter issues:

1. **Check CloudWatch Logs**: The Terraform outputs include links to log groups
2. **Webhook not triggering**: Ensure the webhook URL is correctly configured in GitHub
3. **IAM permissions**: Verify the IAM permissions are sufficient

## Cleanup

When you no longer need the runners:

**On Windows:**
```powershell
.\cleanup.ps1
```

**On Linux/macOS:**
```bash
./cleanup.sh
```

Or manually:
```bash
terraform destroy
```

Don't forget to also remove the webhook URL from your GitHub App settings. 