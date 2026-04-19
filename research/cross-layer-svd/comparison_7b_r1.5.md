# bench_compare: base vs factored

- **Base model:** `Qwen/Qwen2.5-7B` (base)
- **Factored dir:** `factored_out_7b_r1.5` (factored)
- **Device / dtype:** cuda / float16
- **max_new_tokens:** 80, **temperature:** 0.0
- **Prompts compared:** 15

## Speed

| metric | base | factored | delta |
|---|---|---|---|
| Mean tok/s | 4.7 | 4.7 | 4.7 tok/s → 4.7 tok/s (-0.4% ~) |
| Median tok/s | 4.7 | 4.7 | 4.7 tok/s → 4.7 tok/s (-0.3% ~) |
| Mean TTFT | 5 ms | 11 ms | 5 ms → 11 ms (+128.7% ✗) |
| Total bench time | 246.9 s | 245.6 s | 246.9 s → 245.6 s (-0.5% ✓) |
| Model load | 8.2 s | 224.4 s | 8.2 s → 224.4 s (+2627.4% ✗) |

## Resource utilization (during bench)

| metric | base | factored | delta |
|---|---|---|---|
| VRAM peak (torch) | 15321.4 MB | 15321.4 MB | 15321.4 MB → 15321.4 MB (+0.0% ~) |
| VRAM peak (sampled) | 15320.7 MB | 15321.0 MB | 15320.7 MB → 15321.0 MB (+0.0% ~) |
| VRAM mean | 15313.4 MB | 15313.2 MB | 15313.4 MB → 15313.2 MB (-0.0% ✓) |
| RSS peak | 20775.8 MB | 2776.4 MB | 20775.8 MB → 2776.4 MB (-86.6% ✓) |
| RSS mean | 20772.2 MB | 83.0 MB | 20772.2 MB → 83.0 MB (-99.6% ✓) |
| CPU peak % | 215.2% | 209.2% | 215.2% → 209.2% (-2.8% ✓) |
| CPU mean % | 101.3% | 100.2% | 101.3% → 100.2% (-1.1% ✓) |

## Accuracy (automated)

### Exact-match (factual prompts)

| id | expected | base answer | match | factored answer | match |
|---|---|---|---|---|---|
| `fact_cap_germany` | Berlin | 'Berlin. It is the largest city in Germa' | ✓ | 'Berlin. It is the largest city in Germa' | ✓ |
| `fact_cap_japan` | Tokyo | 'Tokyo, which is also the largest city i' | ✓ | 'Tokyo, which is also the largest city i' | ✓ |
| `fact_cap_brazil` | Brasilia | '____\nA. Brasilia\nB. Rio de Janeiro\nC. S' | ✓ | '____\nA. Brasilia\nB. Rio de Janeiro\nC. S' | ✓ |
| `fact_math` | 40 | 'what number?\nTo find the sum of 17 and' | ✗ | 'what number?\nTo find the sum of 17 and' | ✓ |
| `fact_year` | 1989 | '____.\nA. 1989\nB. 1990\nC. 1991\nD. 1992\nA' | ✓ | '____.\nA. 1989\nB. 1990\nC. 1991\nD. 1992\nA' | ✓ |

**Exact-match score:** base **4/5**, factored **5/5**

### Greedy token agreement (first N tokens identical)

Percent of leading tokens where factored's argmax equals base's, stopping at first divergence. 100% = the two models are indistinguishable greedy-decoding on this prompt.

| id | type | base tokens | matches | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 67 | 84% |
| `fact_cap_japan` | factual | 80 | 17 | 21% |
| `fact_cap_brazil` | factual | 80 | 59 | 74% |
| `fact_math` | factual | 80 | 18 | 22% |
| `fact_year` | factual | 80 | 80 | 100% |
| `comp_fox` | completion | 77 | 2 | 3% |
| `comp_road` | completion | 80 | 80 | 100% |
| `comp_storm` | completion | 80 | 12 | 15% |
| `reason_syllogism` | reasoning | 80 | 62 | 78% |
| `reason_inverse` | reasoning | 80 | 80 | 100% |
| `reason_arithmetic` | reasoning | 61 | 2 | 3% |
| `code_factorial` | code | 62 | 24 | 39% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 41 | 51% |
| `summary_gravity` | summary | 80 | 47 | 59% |

**Overall greedy agreement:** 652/1141 = **57.1%**

By prompt type:

- **code**: 85/123 = 69.1%
- **completion**: 94/237 = 39.7%
- **factual**: 241/400 = 60.2%
- **reasoning**: 144/221 = 65.2%
- **summary**: 88/160 = 55.0%

### Top-5 overlap (base's next token present in factored's top-5)

Looser than greedy match — measures whether the factored model considers base's choice a high-probability option even when its own argmax differs.

| id | type | positions | in top-5 | % |
|---|---|---|---|---|
| `fact_cap_germany` | factual | 80 | 72 | 90% |
| `fact_cap_japan` | factual | 80 | 20 | 25% |
| `fact_cap_brazil` | factual | 80 | 64 | 80% |
| `fact_math` | factual | 80 | 27 | 34% |
| `fact_year` | factual | 80 | 80 | 100% |
| `comp_fox` | completion | 77 | 45 | 58% |
| `comp_road` | completion | 80 | 80 | 100% |
| `comp_storm` | completion | 80 | 18 | 22% |
| `reason_syllogism` | reasoning | 80 | 63 | 79% |
| `reason_inverse` | reasoning | 80 | 80 | 100% |
| `reason_arithmetic` | reasoning | 61 | 60 | 98% |
| `code_factorial` | code | 62 | 32 | 52% |
| `code_fizzbuzz` | code | 61 | 61 | 100% |
| `summary_photosynthesis` | summary | 80 | 44 | 55% |
| `summary_gravity` | summary | 80 | 54 | 68% |

**Overall top-5 overlap:** 800/1141 = **70.1%**

## Side-by-side responses (qualitative)

Paste any of these into the chat for qualitative judgment.

### `fact_cap_germany` (factual)

**Prompt:** The capital of Germany is
  *(expected: Berlin)*

**Base:** ' Berlin. It is the largest city in Germany and the 11th largest city in the European Union. The city is the centre of politics, administration, media, science and culture in Germany. Berlin is the second most populous city in the European Union after London. The city is also a major tourist destination, with over 13 million tourists arriving in 2013. Berlin is'

**Factored:** ' Berlin. It is the largest city in Germany and the 11th largest city in the European Union. The city is the centre of politics, administration, media, science and culture in Germany. Berlin is the second most populous city in the European Union after London. The city is also a major tourist destination, with over 10 million tourists visiting the city every year. The city is home'

### `fact_cap_japan` (factual)

**Prompt:** The capital of Japan is
  *(expected: Tokyo)*

**Base:** ' Tokyo, which is also the largest city in the country. Tokyo is located on the eastern coast of the main island of Honshu and is the political, economic, and cultural center of Japan. It is also one of the most populous cities in the world, with a population of over 13 million people. Tokyo is known for its modern architecture, bustling streets, and vibrant culture. It'

**Factored:** ' Tokyo, which is also the largest city in the country. Tokyo is located on the island of Honshu and is divided into 23 special wards, 26 cities, 5 towns, and 8 villages. The city is known for its vibrant culture, modern architecture, and technological advancements. It is also home to many famous landmarks such as the Tokyo Tower, the Imperial Palace,'

### `fact_cap_brazil` (factual)

**Prompt:** The capital of Brazil is
  *(expected: Brasilia)*

**Base:** " ____\nA. Brasilia\nB. Rio de Janeiro\nC. Sao Paulo\nD. Belo Horizonte\nAnswer:\nA\n\nThe main reason for the formation of the 'suspended river' in the lower reaches of the Yellow River is ____\nA. The Yellow River carries a large amount of silt, and the riverbed is constantly rising.\nB. The Yellow River"

**Factored:** " ____\nA. Brasilia\nB. Rio de Janeiro\nC. Sao Paulo\nD. Belo Horizonte\nAnswer:\nA\n\nThe main reason for the formation of the 'suspended river' in the lower reaches of the Yellow River is ____\nA. The Yellow River flows through the Loess Plateau, and the soil is loose.\nB. The Yellow River flows through"

### `fact_math` (factual)

**Prompt:** Seventeen plus twenty-three equals
  *(expected: 40)*

**Base:** ' what number?\nTo find the sum of 17 and 23, we can break it down into simpler steps. First, we add the tens place digits together and then the ones place digits together, and finally combine the results.\n\n1. Add the tens place digits: \\(10 + 20 = 30\\).\n2. Add the ones place digits: \\(7 +'

**Factored:** " what number?\nTo find the sum of 17 and 23, we can add the numbers step by step. Here's the process:\n\n1. Start with the number 17.\n2. Add 20 to 17. This gives us 37.\n3. Now, add the remaining 3 to 37. This gives us 40.\n\nSo"

### `fact_year` (factual)

**Prompt:** The year the Berlin Wall fell was
  *(expected: 1989)*

**Base:** " ____.\nA. 1989\nB. 1990\nC. 1991\nD. 1992\nAnswer:\nA\n\nThe main reason for the formation of the 'three mountains and two basins' topography in Xinjiang is ____\nA. The result of the uplift of the Earth's crust\nB. The result of"

**Factored:** " ____.\nA. 1989\nB. 1990\nC. 1991\nD. 1992\nAnswer:\nA\n\nThe main reason for the formation of the 'three mountains and two basins' topography in Xinjiang is ____\nA. The result of the uplift of the Earth's crust\nB. The result of"

### `comp_fox` (completion)

**Prompt:** The quick brown fox jumps over the

**Base:** ' lazy dog\ndef count_vowels(string):\n    vowels = [\'a\', \'e\', \'i\', \'o\', \'u\']\n    count = 0\n    for char in string:\n        if char.lower() in vowels:\n            count += 1\n    return count\n\nstring = "The quick brown fox jumps over the lazy dog"\nprint(count_vowels(string))'

**Factored:** ' lazy dog.\ndef count_vowels(string):\n    vowels = [\'a\', \'e\', \'i\', \'o\', \'u\']\n    count = 0\n    for char in string:\n        if char in vowels:\n            count += 1\n    return count\n\nstring = "The quick brown fox jumps over the lazy dog."\nprint(count_vowels(string))'

### `comp_road` (completion)

**Prompt:** Two roads diverged in a yellow wood, and

**Base:** ' sorry I could not travel both And be one traveler, long I stood And looked down one as far as I could To where it bent in the undergrowth; Then took the other, as just as fair, And having perhaps the better claim, Because it was grassy and wanted wear; Though as for that the passing there Had worn them really about the same, And both that morning equally lay In'

**Factored:** ' sorry I could not travel both And be one traveler, long I stood And looked down one as far as I could To where it bent in the undergrowth; Then took the other, as just as fair, And having perhaps the better claim, Because it was grassy and wanted wear; Though as for that the passing there Had worn them really about the same, And both that morning equally lay In'

### `comp_storm` (completion)

**Prompt:** It was a dark and stormy night when

**Base:** " I was awakened by a loud noise. I got up and looked out of the window. It was raining heavily. I went back to bed and tried to go to sleep again. But I couldn't. I was too excited. I got up and turned on the radio. It was 12 o'clock. The radio said that a big tree had fallen down in the street in front of my"

**Factored:** ' I was awakened by a loud noise. I got up and went to the window. It was raining heavily and the wind was blowing strongly. I looked out of the window and saw a figure running towards the house. I was curious and decided to investigate.\nAs I opened the door, I saw a man standing there. He was soaked in rain and looked very scared. He told me that he had'

### `reason_syllogism` (reasoning)

**Prompt:** All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is

**Base:** ' warm-blooded. This reasoning is:\nA. Correct\nB. Incorrect\nAnswer:\nA\n\nWhich of the following statements about the characteristics of the Chinese revolution is incorrect?\nA. The Chinese revolution was against imperialism and feudalism, not capitalism or the national bourgeoisie in general.\nB. The Chinese revolution was a peasant revolution led by the Communist Party of China, not a proletarian revol'

**Factored:** ' warm-blooded. This reasoning is:\nA. Correct\nB. Incorrect\nAnswer:\nA\n\nWhich of the following statements about the characteristics of the Chinese revolution is incorrect?\nA. The Chinese revolution was against imperialism and feudalism, not capitalism or the national bourgeoisie in general.\nB. The Chinese revolution was led by the Communist Party of China, not the bourgeoisie.\nC. The Chinese revoluti'

### `reason_inverse` (reasoning)

**Prompt:** If it is raining, then the ground is wet. The ground is not wet. Therefore,

**Base:** ' it is not raining. This is an example of which logical fallacy?\nThe logical fallacy in the given statement is "Denying the antecedent." This fallacy occurs when someone assumes that if the antecedent (the "if" part of the conditional statement) is false, then the consequent (the "then" part of the conditional statement) must also be false. In'

**Factored:** ' it is not raining. This is an example of which logical fallacy?\nThe logical fallacy in the given statement is "Denying the antecedent." This fallacy occurs when someone assumes that if the antecedent (the "if" part of the conditional statement) is false, then the consequent (the "then" part of the conditional statement) must also be false. In'

### `reason_arithmetic` (reasoning)

**Prompt:** Alice has 3 apples. Bob gives her 5 more apples, then she eats 2. Alice now has

**Base:** ' _____ apples.\nAlice starts with 3 apples. Bob gives her 5 more apples, so she now has 3 + 5 = 8 apples. Then she eats 2 apples, so she now has 8 - 2 = 6 apples. Alice now has 6 apples.'

**Factored:** ' _____ apples. Alice starts with 3 apples. Bob gives her 5 more apples, so she now has 3 + 5 = 8 apples. Then she eats 2 apples, so she now has 8 - 2 = 6 apples. Alice now has 6 apples.'

### `code_factorial` (code)

**Prompt:** def factorial(n):
    """Return n! using recursion."""
    

**Base:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef fibonacci(n):\n    """Return the nth Fibonacci number using recursion."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef binary_search'

**Factored:** ' if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\ndef factorial(n):\n    """Return n! using iteration."""\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result'

### `code_fizzbuzz` (code)

**Prompt:** # Print FizzBuzz for numbers 1 to 15
for i in range(1, 16):
    

**Base:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

**Factored:** ' if i % 3 == 0 and i % 5 == 0:\n         print("FizzBuzz")\n     elif i % 3 == 0:\n         print("Fizz")\n     elif i % 5 == 0:\n         print("Buzz")\n     else:\n         print(i)'

### `summary_photosynthesis` (summary)

**Prompt:** Photosynthesis is the biological process by which

**Base:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose or other sugars. This process is essential for life on Earth, as it provides the primary source of energy for most living organisms. The process of photosynthesis can be divided into two main stages: the light-dependent reactions and the light-independent reactions (also known as the Calvin cycle).\n\n'

**Factored:** ' plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose or other sugars. This process is essential for life on Earth, as it provides the primary source of energy for most organisms and is the basis of the food chain. The process of photosynthesis occurs in two stages: the light-dependent reactions and the light-independent reactions (also known as the Cal'

### `summary_gravity` (summary)

**Prompt:** Gravity is the fundamental force that

**Base:** ' holds the universe together. It is the force that keeps the planets in orbit around the sun, and it is the force that keeps us on the ground. But what is gravity, and how does it work?\nGravity is a force that attracts two objects with mass towards each other. The strength of the force depends on the mass of the objects and the distance between them. The more massive the objects,'

**Factored:** ' holds the universe together. It is the force that keeps the planets in orbit around the sun, and it is the force that keeps us on the ground. But what is gravity, and how does it work?\nGravity is a force that is exerted by all objects with mass. The more massive an object is, the stronger its gravitational pull. The gravitational force between two objects is proportional to the product'
