package org.example;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class ForkJoinGeneticKMeans {
    private final List<double[]> data;

    private final int k;
    private final int maxGenerations;
    private final int populationSize;
    private final int paramNum;
    private final double crossoverProbability;
    private final double mutationProbability;
    private final Random random;
    List<int[]> population;
    List<Double> fitness;

    public ForkJoinGeneticKMeans(int k, List<int[]> population, int populationSize, int maxIterations, double crossoverProbability, double mutationProbability, List<double[]> data) {
        this.data = data;
        this.k = k;
        this.maxGenerations = maxIterations;
        this.populationSize = populationSize;
        this.crossoverProbability = crossoverProbability;
        this.mutationProbability = mutationProbability;
        this.random = new Random();
        this.population = population;
        this.paramNum = population.get(0).length;
    }

    public List<int[]> run() {
        int generation = 0;
        while (generation < maxGenerations) {
//            System.out.println("Generation: " + generation + " / " + maxGenerations);
            List<int[]> offspring = crossover(population);
            mutate(offspring);
            population.addAll(offspring);
            trueValueCheck(population);
            fitness = evaluate(population);
            this.select();
            generation++;
        }
        return population;
    }

    public List<Double> evaluate(List<int[]> population) {
        ForkJoinPool forkJoinPool = new ForkJoinPool();
        FitnessCalculatorTask task = new FitnessCalculatorTask(population, 0, population.size());
        return forkJoinPool.invoke(task);
    }

    private double computeFitness(int[] chromosome) {
        List<double[]> centroids = decode(chromosome);
        List<List<double[]>> result = KMeans(data, centroids);
        if (result == null)
            return -1.0;
        return scores.calinskiHarabaszIndex(result);
    }

    public class FitnessCalculatorTask extends RecursiveTask<List<Double>> {
        private static final int THRESHOLD = 10;
        private final List<int[]> population;
        private final int start;
        private final int end;

        public FitnessCalculatorTask(List<int[]> population, int start, int end) {
            this.population = population;
            this.start = start;
            this.end = end;
        }

        @Override
        protected List<Double> compute() {
            int length = end - start;
            if (length <= THRESHOLD) {
                return computeDirectly();
            }

            int mid = (start + end) / 2;
            FitnessCalculatorTask leftTask = new FitnessCalculatorTask(population, start, mid);
            FitnessCalculatorTask rightTask = new FitnessCalculatorTask(population, mid, end);

            invokeAll(leftTask, rightTask);

            List<Double> leftResult = leftTask.join();
            List<Double> rightResult = rightTask.join();

            List<Double> result = new ArrayList<>(leftResult);
            result.addAll(rightResult);

            return result;
        }

        private List<Double> computeDirectly() {
            List<Double> fitness = new ArrayList<>();
            for (int i = start; i < end; i++) {
                int[] chromosome = population.get(i);
                double fit = computeFitness(chromosome);
                fitness.add(fit);
            }
            return fitness;
        }
    }

    private void select() {
        List<int[]> nextGeneration = new ArrayList<>();
        int bestIndex=0;
        double bestFitness=fitness.get(0);
        double sum;
        double[] cumulativeFitness = new double[fitness.size()];
        List<Double> newFitness = new ArrayList<>();
        cumulativeFitness[0] = fitness.get(0);
        // 计算总适应度和累计适应度,保留最优个体
        for (int i = 1; i < fitness.size(); i++) {
            cumulativeFitness[i] = cumulativeFitness[i - 1] + fitness.get(i);
            if (fitness.get(i) > bestFitness) {
                bestFitness = fitness.get(i);
                bestIndex = i;
            }
        }
        nextGeneration.add(population.get(bestIndex));
        newFitness.add(fitness.get(bestIndex));
        sum = cumulativeFitness[cumulativeFitness.length - 1];
        // 选择新一代个体
        while (nextGeneration.size() < populationSize) {
            double r = Math.random() * sum;
            for (int j = 0; j < cumulativeFitness.length; j++) {
                if (r <= cumulativeFitness[j]) {
                    nextGeneration.add(population.get(j));
                    newFitness.add(fitness.get(j));
                    break;
                }
            }
        }
        population = nextGeneration;
        fitness = newFitness;
    }

    private List<int[]> crossover(List<int[]> parents) {
        List<int[]> offspring = new ArrayList<>();
        while (offspring.size() < populationSize / 2) {
            int index1 = random.nextInt(parents.size());
            int index2 = random.nextInt(parents.size());
            if (index1 == index2)
                continue;
            if (random.nextDouble() < crossoverProbability) {
                int[] parent1 = parents.get(index1);
                int[] parent2 = parents.get(index2);
                int[] child1 = Arrays.copyOf(parent1, parent1.length);
                int[] child2 = Arrays.copyOf(parent2, parent2.length);
                int crossoverPoint = random.nextInt(paramNum);
                for (int j = crossoverPoint; j < paramNum; j++) {
                    int temp = child1[j];
                    child1[j] = child2[j];
                    child2[j] = temp;
                }
                offspring.add(child1);
                offspring.add(child2);
            } else {
                offspring.add(parents.get(index1));
                offspring.add(parents.get(index2));
            }
        }
        return offspring;
    }

    private void mutate(List<int[]> offspring) {
        for (int[] chromosome : offspring) {
            for (int i = 0; i < paramNum; i++) {
                if (random.nextDouble() < mutationProbability) {
                    if (chromosome[i] == 0)
                        chromosome[i] = 1;
                    else
                        chromosome[i] = 0;
                }
            }
        }
    }

    private void trueValueCheck(List<int[]> population) {
        for (int[] d : population) {
            int trueCount = countTrueValues(d);
            if (trueCount > k) {
                while (trueCount != k) {
                    int indexToMutate = random.nextInt(paramNum);
                    if (d[indexToMutate] == 1) {
                        d[indexToMutate] = 0;
                        trueCount--;
                    }
                }
            } else {
                while (trueCount != k) {
                    int indexToMutate = random.nextInt(paramNum);
                    if (d[indexToMutate] == 0) {
                        d[indexToMutate] = 1;
                        trueCount++;
                    }
                }
            }
        }
    }

    private int countTrueValues(int[] individual) {
        int count = 0;
        for (int b : individual) {
            count += b;
        }
        return count;
    }

    private List<double[]> decode(int[] chromosome) {
        List<double[]> centroids = new ArrayList<>();
        for (int i = 0; i < chromosome.length; i++) {
            if (chromosome[i] == 1) {
                centroids.add(data.get(i));
            }
        }
        return centroids;
    }

    private List<List<double[]>> KMeans(List<double[]> data, List<double[]> centroids) {
        int iter = 0;
        List<List<double[]>> result = new ArrayList<>();
        while (iter < 10) {
            result.clear();
            for (int i = 0; i < k; i++) {
                result.add(new ArrayList<>());
            }
            for (double[] datum : data) {
                int index = 0;
                double minDis = distance(datum, centroids.get(0));
                for (int i = 1; i < centroids.size(); i++) {
                    double dis = distance(datum, centroids.get(i));
                    if (dis < minDis) {
                        minDis = dis;
                        index = i;
                    }
                }
                result.get(index).add(datum);
            }
            List<double[]> newCenter = new ArrayList<>();
            for (List<double[]> aCluster : result) {
                if (aCluster.size() == 0) {
                    return null;
                }
                double[] centerData = aCluster.get(0).clone();
                for (int j = 1; j < aCluster.size(); j++) {
                    double[] tmp = aCluster.get(j).clone();
                    for (int m = 0; m < tmp.length; m++) {
                        centerData[m] += tmp[m];
                    }
                }
                for (int j = 0; j < centerData.length; j++) {
                    centerData[j] /= aCluster.size();
                }
                newCenter.add(centerData);
            }
            boolean centerChanged = false;
            for (int i = 0; i < k; i++) {
                double[] older = centroids.get(i);
                double[] newer = newCenter.get(i);
                int dataLen = older.length;
                for (int j = 0; j < dataLen; j++) {
                    if (!Objects.equals(newer[j], older[j])) {
                        centerChanged = true;
                        i = k;
                        break;
                    }
                }
            }
            if (!centerChanged)
                break;
            iter++;
            centroids = newCenter;
        }
        return result;
    }

    private double distance(double[] a, double[] b) {
        double distance = 0;
        for (int i = 0; i < a.length; i++) {
            distance += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(distance);
    }
}
