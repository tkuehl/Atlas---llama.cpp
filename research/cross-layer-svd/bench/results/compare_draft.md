# Speculative Decoding Bench — `qwen_draft` vs `baseline`

Baseline `baseline`: 25 prompts  
Condition `qwen_draft`: 25 prompts

## Per-slice summary

| Slice | Base tok/s | Cond tok/s | Speedup | Accept rate | TTFT Δ (ms) |
|---|---:|---:|---:|---:|---:|
| structured_output | 83.7 | 152.3 | 1.82× | 80.1% | -2 |
| code | 89.7 | 163.0 | 1.82× | 81.3% | -2 |
| factual_qa | 94.0 | 138.1 | 1.47× | 76.9% | -3 |
| reasoning | 84.7 | 143.4 | 1.69× | 83.9% | -2 |
| conversational | 84.1 | 84.5 | 1.00× | 55.4% | -1 |

**Overall:** baseline=87.2 tok/s, condition=136.3 tok/s, speedup=1.56×, acceptance=71.5%

## Per-prompt response preview (first 80 chars)

| ID | Slice | Baseline | Condition |
|---|---|---|---|
| code_01 | code | `def factorial(n):     """     Returns the factorial of a non-negative integer n.` | `def factorial(n):     """     Returns the factorial of a non-negative integer n.` |
| code_02 | code | ````c /* Standard C header */ #include <stdio.h>  void reverse(char *s) {     cha` | ````c /* Standard C header */ #include <stdio.h>  void reverse(char *s) {     cha` |
| code_03 | code | ````python def is_prime(n):     if n < 2:         return False     for i in range` | ````python def is_prime(n):     if n < 2:         return False     for i in range` |
| code_04 | code | `func Sum(nums []int) int { 	for _, num := range nums { 		sum += num 	} 	return s` | `func Sum(nums []int) int { 	for _, num := range nums { 		sum += num 	} 	return s` |
| code_05 | code | ````python def fibonacci(n):     a, b = 0, 1     for _ in range(n):         a, b ` | ````python def fibonacci(n):     a, b = 0, 1     for _ in range(n):         a, b ` |
| conv_01 | conversational | `As an AI, I don't experience weather or have personal preferences. However, I ca` | `As an AI, I don't experience weather or have personal preferences. However, I ca` |
| conv_02 | conversational | `One interesting hobby that someone new to it might enjoy picking up is **birdwat` | `One interesting hobby that someone new to it might enjoy picking up is **birdwat` |
| conv_03 | conversational | `Starting a new job is an exciting and sometimes overwhelming experience. Here ar` | `Starting a new job is an exciting and sometimes overwhelming experience. Here ar` |
| conv_04 | conversational | `A perfect weekend is a harmonious blend of relaxation, creativity, connection, a` | `A perfect weekend is a harmonious blend of relaxation, creativity, connection, a` |
| conv_05 | conversational | `A good long-term friendship is built on a foundation of trust, mutual respect, a` | `A good long-term friendship is built on a foundation of trust, mutual respect, a` |
| fact_01 | factual_qa | `The capital city of Australia is Canberra.` | `The capital city of Australia is Canberra.` |
| fact_02 | factual_qa | `The novel *1984* was written by George Orwell.` | `The novel *1984* was written by George Orwell.` |
| fact_03 | factual_qa | `The chemical symbol for gold is **Au**.` | `The chemical symbol for gold is **Au**.` |
| fact_04 | factual_qa | `Humans first landed on the Moon in 1969.` | `Humans first landed on the Moon in 1969.` |
| fact_05 | factual_qa | `The largest ocean on Earth is the Pacific Ocean.` | `The largest ocean on Earth is the Pacific Ocean.` |
| reason_01 | reasoning | `Let's solve this step by step:  1. **Start with the total number of apples:**   ` | `Let's solve this step by step:  1. **Start with the total number of apples:**   ` |
| reason_02 | reasoning | `We are given:  - Train A leaves **Station A** at **9:00 AM** traveling at **60 m` | `We are given:  - Train A leaves **Station A** at **9:00 AM** traveling at **60 m` |
| reason_03 | reasoning | `We are given a recipe that makes **6 cookies** using **2 cups of flour**. We wan` | `We are given a recipe that makes **6 cookies** using **2 cups of flour**. We wan` |
| reason_04 | reasoning | `To find the cost of fencing the entire perimeter of the rectangular garden, we n` | `To find the cost of fencing the entire perimeter of the rectangular garden, we n` |
| reason_05 | reasoning | `To find out how much money the child will have saved after 16 weeks, we can foll` | `To find out how much money the child will have saved after 16 weeks, we can foll` |
| struct_01 | structured_output | `{   "id": 1,   "name": "John Doe",   "email": "john.doe@example.com",   "age": 3` | `{   "id": 1,   "name": "John Doe",   "email": "john.doe@example.com",   "age": 3` |
| struct_02 | structured_output | `{   "port": 8080,   "host": "0.0.0.0",   "max_connections": 100,   "timeout_seco` | `{   "port": 8080,   "host": "0.0.0.0",   "max_connections": 100,   "timeout_seco` |
| struct_03 | structured_output | `[{"id":1,"name":"Wireless Bluetooth Earbuds","price":29.99,"in_stock":true,"cate` | `[{"id":1,"name":"Wireless Bluetooth Earbuds","price":29.99,"in_stock":true,"cate` |
| struct_04 | structured_output | `CREATE TABLE users (     id INTEGER PRIMARY KEY,     username VARCHAR(50) NOT NU` | `CREATE TABLE users (     id INTEGER PRIMARY KEY,     username VARCHAR(50) NOT NU` |
| struct_05 | structured_output | `{   "level": "INFO",   "handlers": ["console", "file"],   "format": "%(asctime)s` | `{   "level": "INFO",   "handlers": ["console", "file"],   "format": "%(asctime)s` |
