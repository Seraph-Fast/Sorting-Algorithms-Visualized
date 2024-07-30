import os
#import imageio
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import deque 
import math
import pandas as pd
import random
import imageio.v2 as imageio


sSort="Ω(n^2)  θ(n^2)  O(n^2)"
bSort	="Ω(n)	θ(n^2)	O(n^2)"
iSort=	"Ω(n)	θ(n^2)	O(n^2)"
hSort=	"Ω(n log(n))  θ(n log(n))  O(n log(n))"
qSort=	"Ω(n log(n))  θ(n log(n))  O(n^2)"
mSort=	"Ω(n log(n))  θ(n log(n))  O(n log(n))"
bucSort="Ω(n +k)  θ(n +k)  O(n^2)"
rSort="Ω(nk)  θ(nk)  O(nk)"
cSort=	"Ω(n +k)  θ(n +k)  O(n +k)"




class sort(object):
	def __init__(self):
		self.img = []
		self.fig = plt.figure()
		plt.ion()
		self.count = 0
		self.path = 'sp/'
		if not os.path.lexists(self.path):
			os.mkdir(self.path)

	def merge(self,arr, start, mid, end):
		start2 = mid + 1
	
		# If the direct merge is already sorted
		if (arr[mid] <= arr[start2]):
			return
	
		# Two pointers to maintain start
		# of both arrays to merge
		while (start <= mid and start2 <= end):
	
			# If element 1 is in right place
			if (arr[start] <= arr[start2]):
				start += 1
			else:
				value = arr[start2]
				index = start2
	
				# Shift all the elements between element 1
				# element 2, right by 1.
				while (index != start):
					arr[index] = arr[index - 1]
					index -= 1
	
				arr[start] = value
	
				# Update all the pointers
				start += 1
				mid += 1
				start2 += 1
		self.draw(arr, fun = 'MERGE SORT',text=mSort)

  

  
  
	def merge_sort(self,arr, l, r):
		if (l < r):
	
			# Same as (l + r) / 2, but avoids overflow
			# for large l and r
			m = l + (r - l) // 2
	
			# Sort first and second halves
			self.merge_sort(arr, l, m)
			self.merge_sort(arr, m + 1, r)
	
			self.merge(arr, l, m, r)


	def quick_sort(self, s, a, b):
		if a >= b:
			return
		base = s[b]
		left, right = a, b - 1
		while left <= right:
			while left <= right and s[left] < base:
				left += 1
			while left <= right and s[right] > base:
				right -= 1
			if left <= right:
				s[left], s[right] = s[right], s[left]
				left, right = left + 1, right - 1
		s[left], s[b] = s[b], s[left]
		if left != b:
			self.draw(s, fun = 'QUICK SORT',text=qSort)
		self.quick_sort(s, a, left - 1)
		self.quick_sort(s, left + 1, b)


	def search(self, s, i):
		index, value = i, s[i]
		for j in range(i, len(s)):
			if s[j] < value:
				index, value = j, s[j]
		s[index], s[i] = s[i], s[index]
		return index != i

	def heap_adjust(self, s, m, length):
		j = 2 * m + 1
		# since index begins from 0, childs of father is 2*x+1 and 2*x+2
		if j == length - 1:
			if s[m] < s[j]:
				s[m], s[j] = s[j], s[m]
		while j <  length - 1:
			if s[j] < s[j + 1]:
				j += 1
			if s[m] > s[j]:
				break
			s[m], s[j], m, j= s[j], s[m], j, 2 * j + 1

	def heap_sort(self, s):
		i = len(s) // 2 - 1
		while i >= 0:
			self.heap_adjust(s, i, len(s))
			i -= 1
		i = len(s)
		while i > 0:
			s[0], s[i - 1], i = s[i - 1], s[0], i - 1
			self.draw(s, fun = 'HEAP SORT',text=hSort)
			self.heap_adjust(s, 0, i)

	def insert_sort(self, s):
		i = j = len(s)
		while i > 0:
			if self.insert_next(s, j - i):
				self.draw(s, fun = 'INSERT SORT',text=iSort)
			i -= 1

	def insert_next(self, s, k):
		index, value = k, s[k]
		for i in range(k, len(s)):
			if s[i] < value:
				index, value = i, s[i]
		s[k], s[index] = s[index], s[k]
		return index != k

	def bubble_sort(self, s):
		n = len(s)
		for j in range(n - 1):
			for i in range(n - j - 1):
				if s[i] > s[i + 1]:
					s[i], s[i + 1] = s[i + 1], s[i]
					self.draw(s, fun = 'BUBBLE SORT',text=bSort)


	def radix_sort(self,arr):
		
		# Determine the maximum number of digits
		max_digits = len(str(max(arr)))

		# Perform counting sort for each digit starting from the least significant digit
		for i in range(max_digits):
			# Create a list of queues, one for each digit (0-9)
			digit_queues = [deque() for _ in range(10)]

			# For each number in the input list, put it in the corresponding queue
			# based on the digit at the current position
			for num in arr:
				digit = (num // (10 ** i)) % 10
				digit_queues[digit].append(num)

			# Concatenate all the queues to get the sorted list for the current digit
			arr = []
			for q in digit_queues:
				arr.extend(q)
			
			self.draw(arr, fun = 'radix_sort',text=rSort)
			plt.pause(1)
		
		plt.pause(5)


	
	def count_sort(self,arr):
		maxval = max(arr)+1
		size = len(arr)
		output = [0] * size

		# count array initialization
		count = [0] * maxval

		# storing the count of each element 
		for m in range(0, size):
			count[arr[m]] += 1

		# storing the cumulative count
		for m in range(1, maxval):
			count[m] += count[m - 1]

		# place the elements in output array after finding the index of each element of original array in count array
		m = size - 1
		while m >= 0:
			output[count[arr[m]] - 1] = arr[m]
			count[arr[m]] -= 1
			m -= 1

		for m in range(0, size):
			arr[m] = output[m]
			self.draw(arr, fun = 'count_sort',text=cSort)
		
		plt.pause(5)


	def hybrid_sort(self,numbers, low, high, k):
		self.draw(numbers, fun = 'hybird_sort')

		if low < high:
			# Partition the list around the pivot
			pivotIndex = self.partition(numbers, low, high)

			# Recursively sort the left and right partitions, but use insertion sort
			# for subarrays with fewer than k elements
			if pivotIndex - low < k:
				self.insertion_sort(numbers, low, pivotIndex)
			else:
				self.hybrid_sort(numbers, low, pivotIndex - 1, k)

			if high - pivotIndex < k:
				self.insertion_sort(numbers, pivotIndex, high)
			else:
				self.hybrid_sort(numbers, pivotIndex + 1, high, k)

	def partition(self,numbers, low, high):
		pivot = numbers[high]
		i = low

		for j in range(low, high):
			if numbers[j] <= pivot:
				temp = numbers[i]
				numbers[i] = numbers[j]
				numbers[j] = temp
				i += 1

		temp = numbers[i]
		numbers[i] = numbers[high]
		numbers[high] = temp

		return i

	def insertion_sort(self,numbers, low, high):
		for i in range(low + 1, high + 1):
			key = numbers[i]
			j = i - 1
			while j >= low and numbers[j] > key:
				numbers[j + 1] = numbers[j]
				j -= 1
			numbers[j + 1] = key

	def preprocess(self,numbers, k):
	# Initialize an array to keep track of the number of occurrences
			# of each integer in the range 0 to k
		counts = [0] * (k + 1)

		# Count the number of occurrences of each integer
		for number in numbers:
			counts[number] += 1

		# Preprocess the counts array to make it easier to answer range queries
		for i in range(1, k + 1):
			counts[i] += counts[i - 1]

		return counts

	def count_range(self,counts, a, b):
		# If a is greater than or equal to 1, subtract the number of occurrences
		# of integers less than a from the total number of occurrences in the range
		if a >= 1:
			return counts[b] - counts[a - 1]
		
		return counts[b]

	def insertionSort(self,b):
		for i in range(1, len(b)):
			up = b[i]
			j = i - 1
			while j >= 0 and b[j] > up: 
				b[j + 1] = b[j]
				j -= 1
			b[j + 1] = up     
		return b     
              
	def bucket_sort(self,x):
		arr = []
		slot_num = 10 
		for i in range(slot_num):
			arr.append([])
			
		# Put array elements in different buckets 
		for j in x:
			index_b = int(slot_num * j) 
			arr[index_b].append(j)
		
		# Sort individual buckets 
		for i in range(slot_num):
			arr[i] = self.insertionSort(arr[i])
			flatt=sum(arr, [])
			self.draw(flatt, fun = 'Bucket Sort',text=bucSort)

			
		k = 0
		for i in range(slot_num):
			for j in range(len(arr[i])):
				x[k] = arr[i][j]
				k += 1
				self.draw(x, fun = 'Bucket Sort',text=bucSort)

		return x


	def draw(self, s, fun,text=""):
		if len(s):
			
			self.count += 1
			plt.clf()
			bars = plt.bar(range(len(s)), s, width = 0.3, color = 'purple')
			plt.title(fun)
			if(text!=""):
				plt.text(5, 610,"time complexity: "+text)
			for bar in bars:
				yval = bar.get_height()
				plt.text(bar.get_x(), yval + .01, yval)
			plt.savefig('sp/' + str(self.count))
			plt.pause(0.5)


	def get_gif(self):
		name = os.listdir(self.path)
		name.sort(key=lambda x:int(x[:-4]))
		for ele in name:
			path = self.path + ele
			self.img.append(imageio.imread(path))
		imageio.mimsave('sort.gif', self.img, fps = 8)

def Algorithm(algo = "Bucket Sort"):
# Open the CSV file
	with open('number.csv', 'r') as csvfile:
	# Create a CSV reader object
		reader = csv.reader(csvfile)

		# Initialize a list to store the integers
		integers = []

		# Iterate over the rows in the CSV file
		for row in reader:
			# Iterate over the columns in the row
			for col in row:
			# Check if the column value is an integer
				if col.isdigit():
					# If it is, add it to the list of integers
					integers.append(int(col))
        
	
	s = integers
	p = sort()

	if(algo == 'Bubble Sort'):
		p.bubble_sort(list(s))
	elif(algo == 'Insertion Sort'):
		p.insert_sort(list(s))

	elif(algo == 'Merge Sort'):
		p.merge_sort(s,0,len(s)-1)
		print(s)

	elif(algo == 'Quick Sort'):
		p.quick_sort(list(s),0,len(s)-1)

	elif(algo == 'Heap Sort'):
		p.heap_sort(list(s))

	elif(algo == 'Bucket Sort'):
		s = [round(random.uniform(0., 1.),2) for i in range(10)]
		ans = p.bucket_sort(s)
		p.draw(ans, fun = 'Bucket Sort')

	elif(algo == 'Radix Sort'):
		p.radix_sort(list(s))

	elif(algo == '8.2.4 Algo'):
		numofins = p.preprocess(s,max(s))
		print(p.count_range(numofins,10,50))

	elif(algo == '7.4.5 Sort'):
		p.hybrid_sort(s,0,len(s)-1,int(math.sqrt(len(s))))
		p.draw(s, fun = 'hybird_sort Done')
		plt.pause(5)
	elif(algo == 'Count Sort'):
		p.count_sort(list(s))
	
	p.get_gif()



