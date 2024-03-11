import time
import matplotlib.pyplot as plt

import numpy as np

def generate_random_data(size):
    return np.random.randint(1, 1000, size=size)
    #Sorted Arrays
    #return np.arange(1, size + 1)

    #Reverse Sorted Arrays
    #return np.arange(size, 0, -1)

    # Arrays with Few Unique Elements
    # unique_elements = np.random.choice(10, size=3)
    # return np.random.choice(unique_elements, size=size)

    #Arrays with a Single Unique Element
    #return np.full(size, fill_value=1337)

    #Sparse Arrays
    # sparse_ratio = 0.9
    # return np.random.choice([0, 1], size=size, p=[sparse_ratio, 1 - sparse_ratio])


    #Arrays with Pre-sorted Blocks
    # size+=10
    # block_size = 10
    # num_blocks = size // block_size
    # return np.concatenate([np.arange(i, i + block_size) for i in range(0, num_blocks * block_size, block_size)])


def measure_execution_time(sort_function, data):
    start_time = time.time()
    sort_function(data)
    end_time = time.time()
    return end_time - start_time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i, j, k = 0, 0, 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def counting_sort(arr):
    max_value = max(arr)
    count = [0] * (max_value + 1)

    for num in arr:
        count[num] += 1

    output = []
    for i in range(len(count)):
        output.extend([i] * count[i])

    return output

def shaker_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while (swapped and start < end):
        swapped = False

        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        start += 1


def run_experiment(algorithm, data_sizes, repetitions=5):
    plot_results = []  
    scatter_results = [] 

    for size in data_sizes[algorithm]:
        execution_times = []
        for _ in range(repetitions):
            
            random_data = generate_random_data(size)

            execution_time = measure_execution_time(algorithm, random_data)

            execution_times.append(execution_time)

        plot_results.append((size, np.mean(execution_times)))

        scatter_results.append((size, execution_times))

        print(algorithm, size, execution_times)

    return plot_results, scatter_results

def plot_and_scatter_results(algorithms, data_sizes):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for algorithm in algorithms:
        plot_results, scatter_results = run_experiment(algorithm, data_sizes)
        plot_results.sort(key=lambda x: x[0])
        sizes, average_times = zip(*plot_results)

        color = plt.cm.get_cmap('tab10')(algorithms.index(algorithm) / len(algorithms))

        ax1.plot(sizes, average_times, label=f'{algorithm.__name__} - Mean', color=color)

        all_sizes = []
        all_execution_times = []
        for scatter_result in scatter_results:
            sizes, execution_times = scatter_result
            all_sizes.extend([sizes] * len(execution_times))
            all_execution_times.extend(execution_times)

        label = f'{algorithm.__name__}'
        ax2.scatter(all_sizes, all_execution_times, label=label, color=color, alpha=0.5)

    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Mean Execution Time (seconds)')
    ax1.legend()
    fig1.show()

    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.legend()
    fig2.show()




if __name__ == '__main__':

    data_sizes = {
        bubble_sort: list(range(1,5002,1000)), #n^2
        insertion_sort: list(range(1,5002,1000)), #n^2
        selection_sort: list(range(1,5002,1000)), # n^2
        shaker_sort: list(range(1,5002,1000)), # n^2
        merge_sort: list(range(1,100002,5000)),
        quick_sort: list(range(1,100002,5000)),
        counting_sort: list(range(1,100002,5000)),
        
    }
    algorithms = [bubble_sort,insertion_sort,selection_sort,shaker_sort,merge_sort,quick_sort,counting_sort] 
    
    #algorithms = [counting_sort,merge_sort,quick_sort]
    plot_and_scatter_results(algorithms, data_sizes)
    plt.show()
