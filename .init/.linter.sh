#!/bin/bash
cd /home/kavia/workspace/code-generation/meme-creator-platform-207065-207074/meme_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

