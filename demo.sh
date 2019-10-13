#!/usr/bin/bash
python main.py --step1 --learning-rate 0.001
python main.py --step2 --resume --learning-rate 0.0001
