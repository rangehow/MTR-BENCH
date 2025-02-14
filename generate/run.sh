#!/bin/bash

export OPENAI_API_KEY="1790715889671905303"
export OPENAI_BASE_URL="https://aigc.sankuai.com/v1/openai/native/"

export INPUT_FILE="original_data/non_fuhao_style_filtered_output2.json"
export OUTPUT_FILE="original_data/output_gpt4o_results.jsonl"
export CACHE_DIR="cache"

mkdir -p "$CACHE_DIR"
python conversation_generator.py