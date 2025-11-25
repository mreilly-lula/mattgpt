import fs from "node:fs";
import {
  one_hot,
  randFloat,
  sum,
  weightedRandomSample,
  zip,
} from "../matt-torch";
import { Tensor } from "../structures/Tensor";

const SPECIAL = ".";
const bigramKey = (a: string, b: string): string => `${a}:${b}`;
const bigramVals = (key: string): string[] => key.split(":");
const names = fs.readFileSync("./names.txt", "utf8");
const words = names.split("\n");

console.log(words.length);
// // build map of bigrams and their frequencies
const bigramCounter = new Map<string, number>();
for (const word of words) {
  const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
  const comparator = zip(wordArr, wordArr.slice(1));
  for (const [a, b] of comparator) {
    const key = bigramKey(a, b);
    const val = bigramCounter.get(key) || 0;
    bigramCounter.set(key, val + 1);
  }
}

const uniqChars = [SPECIAL, ...Array.from(new Set(words.join(""))).sort()];
// console.log(uniqChars);
const stoi = (s: string) => uniqChars.indexOf(s);
const itos = (i: number) => uniqChars[i];
const MODEL_SMOOTHING_INT = 1;
// create a table matrix to represent the bigram counter map
// 28 by 28 tensor to represent every char plus start and end
// MODEL SMOOTHING: set fill to 1 instead of 0 to ensure log likelihood is never - infinity
// the more we start with, the smoother the model (this isn't necessarily good, smooth != low loss)
const bigramTensor = new Tensor([uniqChars.length, uniqChars.length] as const, MODEL_SMOOTHING_INT);

[...bigramCounter.entries()].map(([k, v]) => {
  const chars = bigramVals(k);
  const x = stoi(chars[0]);
  const y = stoi(chars[1]);

  bigramTensor.set([x, y], v);
});

// console.table(bigramTensor);
const normalizedTensor = bigramTensor.normalize();
console.log(normalizedTensor)
// console.log(sum(normalizedTensor[0]))

// manually sample a full word from the distribution
for (let iter = 0; iter < 20; iter++) {
  let index = 0;
  let builtStr = [];
  while (true) {
    const normalizedP = normalizedTensor.row([index]).map((item) => item.data);

    index = weightedRandomSample(normalizedP, 1)[0];
    builtStr.push(itos(index));
    if (index === 0) {
      break;
    }
  }

  console.log(builtStr.join(""));
}
let counter = 0;
let log_likelihood = 0.0;
for (const word of words) {
  const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
  const comparator = zip(wordArr, wordArr.slice(1));
  for (const [a, b] of comparator) {
    const idx1 = stoi(a);
    const idx2 = stoi(b);
    log_likelihood += Math.log(normalizedTensor.at([idx1, idx2]).data);
    counter++;
    // console.log(a, b, normalizedTensor[idx1][idx2], Math.log(normalizedTensor[idx1][idx2]))
  }
}

// console.log(log_likelihood)
const neg_log_likelihood = -log_likelihood;
// console.log(neg_log_likelihood / counter)


// GOAL: maximize log likelihood which is the same as minimizing the negative log likelihood
// which is the same as minimizing the average negative log likelihood
// log(a+b+c) = log(a) + log(b) + log(c)

// CREATE TRAINING SET
// console.log(words.length);
// // build map of bigrams and their frequencies
// const xs: number[] = [];
// const ys: number[] = [];
// for (const word of words.slice(0,1)) {
//   const wordArr = [SPECIAL, ...word.split(""), SPECIAL];
//   const comparator = zip(wordArr, wordArr.slice(1));
//   for (const [a, b] of comparator) {
//     xs.push(stoi(a));
//     ys.push(stoi(b));
//   }
// }

// function* fillXs() {
//   let index = 0;
//   while (index < xs.length) {
//     yield xs[index++];
//   }
// }

// function* fillYs() {
//   let index = 0;
//   while (index < ys.length) {
//     yield ys[index++];
//   }
// }

// const xTensor = tensor([xs.length] as const, fillXs())
// console.log(xs)
// console.log(xTensor)
// const yTensor = tensor([ys.length] as const, fillYs());
// console.log(yTensor)

// const oneHot = one_hot(xTensor, uniqChars.length);
// console.log(oneHot);

// const testing = new Tensor([2, 3], 1);
// const test2 = new Tensor([2, 3], () => randFloat(0, 1));

// console.log(testing.shape())
// console.log(testing.dims)
// console.log(testing.size)
// console.log(testing.normalize())

// console.log(test2)
// console.log(test2.dims)
// console.log(test2.at([1,2]))