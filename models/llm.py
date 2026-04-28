import os
import boto3
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# from langchain_aws import ChatBedrockConverse
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage,AIMessage
from langchain_core.embeddings import Embeddings
from const.consts import *
# from langchain_huggingface import HuggingFaceEmbeddings

class LLM:
    def __init__(self, vendor_name: str, temperature: float=0.8, beta_use_converse_api:bool=False):
        self.model = self._initialize_llm(vendor_name, temperature, beta_use_converse_api)
        self.vendor_name = vendor_name

    def _resolve_azure_temperature(self, deployment_name: str, default_temperature: float) -> float:
        deployment = deployment_name.lower()

        if deployment.startswith("gpt-5.1-chat"):
            return 1.0

        if deployment.startswith("gpt-4.1"):
            return 0.8

        return default_temperature

    def _initialize_llm(self, vendor_name: LLM_VENDOR, temperature: float, beta_use_converse_api:bool=False) -> BaseChatModel:
        if vendor_name == LLM_VENDOR.AZURE:
            deployment_name = os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']
            resolved_temperature = self._resolve_azure_temperature(deployment_name, temperature)

            model=AzureChatOpenAI(
                azure_endpoint      = os.environ['CHAT_AZURE_OPENAI_ENDPOINT'],
                api_version         = os.environ["OPENAI_API_VERSION"],
                api_key             = os.environ["CHAT_AZURE_OPENAI_API_KEY"],
                azure_deployment    = deployment_name,
                temperature         = resolved_temperature,
                max_tokens          = LLM_MAX_TOKENS,
                streaming           = True
            )
            
            return model
        # elif vendor_name == LLM_VENDOR.AWS:            
        #     if AWS_THINKING_MODE == "enabled":
        #         temperature = 1.0
        #         model = ChatBedrockConverse(
        #             aws_access_key_id       = os.environ["AWS_ACCESS_KEY_ID"],
        #             aws_secret_access_key   = os.environ["AWS_SECRET_ACCESS_KEY"],
        #             model_id                = os.environ['BEDROCK_MODEL_ID'],
        #             region_name             = os.environ['BEDROCK_REGION_NAME'],
        #             max_tokens              = LLM_MAX_TOKENS,
        #             temperature             = temperature,
        #             additional_model_request_fields={
        #                 "thinking": {"type": AWS_THINKING_MODE, "budget_tokens": LLM_EXTENDED_THINKING_TOKENS},
        #             },
        #         )
        #     else:
        #         model = ChatBedrockConverse(
        #             aws_access_key_id       = os.environ["AWS_ACCESS_KEY_ID"],
        #             aws_secret_access_key   = os.environ["AWS_SECRET_ACCESS_KEY"],
        #             model_id                = os.environ['BEDROCK_MODEL_ID'],
        #             region_name             = os.environ['BEDROCK_REGION_NAME'],
        #             max_tokens              = LLM_MAX_TOKENS,
        #             temperature             = temperature,
        #         )
        #     return model
        else:
            raise ValueError(f"Model name {vendor_name} not supported.")
    
    def invoke(self,prompt: LanguageModelInput)-> BaseMessage:
        # if self.vendor_name == LLM_VENDOR.AWS and AWS_THINKING_MODE == "enabled":
        #     response=self.model.invoke(prompt)            
        #     text_chunk  = "".join([part["text"] for part in response.content_blocks if "text" in part])
        #     return AIMessage(content=text_chunk)
        # else:
            return self.model.invoke(prompt)
    
    def getModel(self):
        return self.model

class Embedding:
    def __init__(self, vendor_name: str):
        self.embeddings = self._initialize_llm(vendor_name)

    def _initialize_llm(self, vendor_name: LLM_VENDOR) -> Embeddings:
        if vendor_name == LLM_VENDOR.AZURE:
            embeddings = AzureOpenAIEmbeddings(
                api_key             = os.environ["EMBEDDINGS_AZURE_OPENAI_API_KEY"],
                azure_endpoint      = os.environ["EMBEDDINGS_AZURE_OPENAI_ENDPOINT"],
                api_version         = os.environ["OPENAI_API_VERSION"],
                azure_deployment    = os.environ["EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME"]
            )
            
            return embeddings
        # elif vendor_name == LLM_VENDOR.AWS:
        #     embeddings = BedrockEmbeddings(
        #         aws_access_key_id       = os.environ["AWS_ACCESS_KEY_ID"],
        #         aws_secret_access_key   = os.environ["AWS_SECRET_ACCESS_KEY"],
        #         region_name             = os.environ["BEDROCK_REGION_NAME"],
        #         model_id                = os.environ['EMBEDDINGS_MODEL_ID']
        #         )
        #     return embeddings
        # elif vendor_name == LLM_VENDOR.HF:
        #     model_name = "intfloat/multilingual-e5-large"
        #     model_path = f"models/{model_name}"
            
        #     embeddings = HuggingFaceEmbeddings(model_name=model_path) # HuggingFaceEmbeddings is also supported
        #     return embeddings
        else:
            raise ValueError(f"Model name {vendor_name} not supported.")
    
    def getEmbedding(self)-> Embeddings:
        return self.embeddings

