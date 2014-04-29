#!/usr/bin/python
import json
import os, sys
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("name")
	parser.add_argument("-b", "--batch_size", default = 10, type=int)
	parser.add_argument("-n", "--epochs", default=0, type=int)
	parser.add_argument("-u", "--min_updates", default=100, type=int)
	parser.add_argument("-x", "--max_updates", default=0, type=int)
	parser.add_argument("-s", "--effect_size", default=0.1, type=float)
	parser.add_argument("-e", "--epsilon", default = 0.05, type=float)
	parser.add_argument("-f", "--fixed_weights", action="store_true")
	parser.add_argument("-g", "--gibbs", default = 1, type=int)
	parser.add_argument("-m", "--momentum", default = 0.0, type=float)
	parser.add_argument("-t", "--hidden_type", default="binary", type=str)
	parser.add_argument("-w", "--hidden_width", default=128, type=int)
	
	args = parser.parse_args(sys.argv[1:])
	
	rbm = vars(args)
	rbm["hidden"] = [{
		"type": rbm["hidden_type"],
		"width": rbm["hidden_width"]
	}]
	
	del rbm["hidden_type"]
	del rbm["hidden_width"]
	
	print(json.dumps(rbm, sort_keys=True, indent=4))
