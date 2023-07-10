import os, base64
from constructs import Construct
from aws_cdk import (
    Aws,
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_s3_deployment as s3deploy,
)
import aws_cdk as cdk

region = Aws.REGION
account = Aws.ACCOUNT_ID

# CDK Stack for
# 1. Create S3
# 2. Create SageMaker Notebook

class GenAiInpaintStack(Stack):
    """
    The SageMaker Notebook is used to deploy the custom model on a SageMaker endpoint and test it.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ## Create S3 bucket
        self.bucket = s3.Bucket(
            self,
            "genAiInPaintingS3",
            bucket_name=f"gai-inpainting-s3-{account}",
            auto_delete_objects=True,
            removal_policy=cdk.RemovalPolicy.DESTROY)

        # Upload the notebook directory to S3
        notebook_directory = 'notebooks'
        zip_file_path = '/tmp/notebooks.zip'

        # Zip the notebook directory
        os.system('zip -r ' + zip_file_path + ' ' + notebook_directory)

        # Upload the zip file to S3
        s3deploy.BucketDeployment(
            self,
            'genAiS3Deploy',
            sources=[s3deploy.Source.asset(zip_file_path)],
            destination_bucket=self.bucket,
            destination_key_prefix='notebooks'
        )

        ## IAM Roles
        # Create role for Notebook instance
        nRole = iam.Role(self, "genAiInPaintingNotebookAccessRole", assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'))
        nRole.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))

        # Add necessary permissions to the role
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["logs:*"], resources=["*"]))
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["sagemaker:Describe*", "sagemaker:*Model*", "sagemaker:*Endpoint*", "sagemaker:*ProcessingJob*"], resources=["*"]))
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["s3:*"], resources=[self.bucket.bucket_arn + "/*"]))
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["ecr:BatchGetImage"], resources=["arn:aws:ecr:::*"]))
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["s3:ListAllMyBuckets", "s3:ListBucket"], resources=["arn:aws:s3:::*"]))
        nRole.add_to_policy(iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=["iam:PassRole"], resources=["arn:aws:iam::*:role/" + nRole.role_name]))

        # Create the lifecycle configuration content
        lifecycle_config_content = """
            #!/bin/bash
            search_prefix="gai-inpainting-s3"
            bucket_name=$(aws s3 ls | awk '{print $3}' | grep "^$search_prefix")
            echo "Buckets found:"
            echo "$bucket_name"
            aws s3 cp "s3://$bucket_name/notebooks/notebooks/" /home/ec2-user/SageMaker/ --recursive
            sudo chmod -R 777 /home/ec2-user/SageMaker
            """.strip()

        # Encode the lifecycle configuration content
        encoded_config_content = base64.b64encode(lifecycle_config_content.encode()).decode()

        # Create the lifecycle configuration
        lifecycle_config = sagemaker.CfnNotebookInstanceLifecycleConfig(
            self,
            "gaiLcConfig",
            notebook_instance_lifecycle_config_name="gaiLcConfig",
            on_start=[sagemaker.CfnNotebookInstanceLifecycleConfig.NotebookInstanceLifecycleHookProperty(content=encoded_config_content)]
        )

        ## Create SageMaker Notebook instances cluster
        nid = 'genAiInPaintingNotebookInstance'
        notebook = sagemaker.CfnNotebookInstance(
            self,
            nid,
            instance_type='ml.c5.2xlarge',
            volume_size_in_gb=16,
            notebook_instance_name=nid,
            role_arn=nRole.role_arn,
            lifecycle_config_name=lifecycle_config.notebook_instance_lifecycle_config_name
        )