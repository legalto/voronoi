__author__ = 'dressag1'
import matplotlib.pyplot as plt

f = open('Release/performance.txt')


#
results_cpu = []
results_gpu = []
num_sites = []

lines = f.readlines()
for line in lines:
    temp = line.rstrip('\n')
    temp = temp.split()
    if temp[0] == '1':
        results_cpu.append(temp[2])
        num_sites.append(temp[1])
    elif temp[0] == '2':
        results_gpu.append(temp[2])
    print temp


results_cpu = map(float, results_cpu)
results_gpu = map(float, results_gpu)
num_sites = map(int, num_sites)

plt.plot(num_sites, results_cpu, 'r-', label='CPU')
plt.plot(num_sites, results_gpu, 'b-', label='GPU')
plt.legend(loc='upper left')

plt.ylabel('Time (seconds)')
plt.xlabel('Number of Sites')
plt.title('Number of Sites vs Time (GPU and CPU)')
plt.show()
