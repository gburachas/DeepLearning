#!/usr/bin/python
import json
import os, sys
import argparse

args = object()

class Serializable(object):
	def to_JSON(self):
		return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class Node(Serializable):
	def __init__(self, rbm, child):
		self.rbm = rbm
		self.children = [child]

class StructNode(Serializable):
	
	def __init__(self, name, hidden_width):
		
		self.rbm = {
			"name" : name
			,"epsilon" : args.epsilon
			,"epochs" : args.epochs
			,"gibbs" : args.gibbs
			,"fixed_weights" : False
			,"momentum" : args.momentum
			,"batch_size" : args.batch_size
			,"hidden" : [{
				"type": "binary",
				"width": hidden_width
			}]
		}

		self.children = []
	
class LeafNode(Serializable):
	def __init__(self, rbm, data):
		self.rbm = rbm
		self.input = data
	
def buildEncoderPath(rbm, widths, level=0):
	if len(widths) == 0:
		return LeafNode(rbm, args.input)
	else:
		struct = StructNode("depth%s" % level, widths.pop() )
		struct.children.append( buildEncoderPath(rbm, widths, level+1) )
		return struct
		
class DBNNode(Serializable):
	def __init__(self, name, rbm, datafile, widths):
		self.name = name
		self.up_down_epochs = args.up_down_epochs
		self.batch_size = args.batch_size
		self.structure = StructNode(name, widths.pop())
		self.structure.children.append( buildEncoderPath(rbm, widths) )
		if args.output:
			self.structure.output = args.output

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("name")
	parser.add_argument("rbm")
	parser.add_argument("input")
	parser.add_argument("widths", type=int, nargs="+", help='Source file(s)')
	parser.add_argument("--autoencoder", default=False, help="Interpolate widths to make an AutoEncoder", action="store_true")
	parser.add_argument("--batch_size", default = 10, type=int)
	parser.add_argument("--epochs", default=10, type=int)
	parser.add_argument("--epsilon", default = 0.005, type=float)
	parser.add_argument("--gibbs", default = 1, type=int)
	parser.add_argument("--momentum", default = 0.0, type=float)
	parser.add_argument("--output", default = "", type=str)
	parser.add_argument("--up_down_epochs" , default = 10, type=int)
	
	args = parser.parse_args(sys.argv[1:])
	
	if args.autoencoder:
		assoc_memwidth = args.widths.pop()
		code_width = args.widths.pop()
		w = args.widths[0] / 2
		while w > code_width:
			args.widths.append(w)
			w /= 2
		args.widths.append(code_width)
		args.widths.append(assoc_memwidth)
	
	dbn = DBNNode(args.name, args.rbm, args.input, args.widths)
	
	print(dbn.to_JSON())
