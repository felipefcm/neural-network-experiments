
import numpy as np
# from network import NeuralNetwork;

print('random', np.random.randn(3, 2))

def func(x):
	print('The value was', x);
	return x;

print('the array is', [ func(a) for a in range(3,5) ]);
# l = [ 1, 2, 3 ];
# print('ok', l);
# print('inv', l[:-1]);

# print('larger is', np.max([1,2,3,4,22,1]))
print('npMaximum is', np.maximum(0, [ [0.2], [-0.7], [-0.3] ]))
