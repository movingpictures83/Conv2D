import PyPluMA
import PyIO
import torch

class Conv2DPlugin:
   def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
   def run(self):
       myInput = torch.load(PyPluMA.prefix()+"/"+self.parameters["inputfile"])
       x = int(self.parameters["x"])
       y = int(self.parameters["y"])
       z = int(self.parameters["z"])
       stride = int(self.parameters["stride"])
       m = torch.nn.Conv2d(x, y, z, stride=stride)
       self.myOutput = m(myInput)

   def output(self, outputfile):
       torch.save(self.myOutput, outputfile)
       

