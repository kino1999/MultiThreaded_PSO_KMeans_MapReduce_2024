package org.example;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class FJPSOKMeans {
    private final int numParticles;
    private final int dimension;
    private List<double[]> velocities;
    private List<double[]> positions;
    private List<Double>fitness;
    private List<double[]> personalBests;
    private double[] personalBestFitness;
    private double[] globalBest;
    private double globalBestFitness;
    private final Random rand = new Random();
    private final int maxIterations;
    private final double w;
    private final double c1;
    private final double c2;
    private final int k;
    private final List<double[]> data;
    private final int dataLength;

    public FJPSOKMeans(int numParticles, int maxIterations, int k, List<double[]> data,double w,double c1,double c2,List<double[]>positions) {
        this.dataLength = data.get(0).length;
        this.numParticles = numParticles;
        this.maxIterations = maxIterations;
        this.k = k;
        this.dimension = dataLength * k;
        this.data = data;
        this.w=w;
        this.c1=c1;
        this.c2=c2;
        this.positions=positions;
        initializeParticles();
    }

    private void initializeParticles() {
        velocities = new ArrayList<>();
        personalBests = new ArrayList<>();
        personalBestFitness = new double[numParticles];
        for (int i = 0; i < numParticles; i++) {
            double[] velocity = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                velocity[j] = rand.nextDouble() * 2 - 1; // Random values between -1 and 1
            }
            personalBests.add(positions.get(i).clone());
            velocities.add(velocity);
        }
        globalBest = positions.get(0).clone();
        globalBestFitness = computeFitness(positions.get(0));
        for (int i = 0; i < positions.size(); i++) {
            double fit = computeFitness(positions.get(i));
            personalBestFitness[i] = fit;
            if (fit > globalBestFitness || globalBestFitness != globalBestFitness) {
                globalBestFitness = fit;
                globalBest = positions.get(i).clone();
            }
        }
    }

    private List<Double> evaluate(List<double[]> positions) {
        ForkJoinPool forkJoinPool = new ForkJoinPool();
        int CPU=Runtime.getRuntime().availableProcessors();
        int THRESHOLD=numParticles/CPU;
        FitnessCalculatorTask task = new FitnessCalculatorTask(positions, 0, numParticles,THRESHOLD);
        return forkJoinPool.invoke(task);
    }
    public double computeFitness(double[] position) {
        List<double[]> centroids = decode(position);
        List<List<double[]>> result = clustering(centroids);
        if (result== null)
            return -1.0;
        double sc = scores.calinskiHarabaszIndex(result);
        return sc;
    }
    public class FitnessCalculatorTask extends RecursiveTask<List<Double>> {
        private final int THRESHOLD;
        private final List<double[]> positions;
        private final int start;
        private final int end;

        public FitnessCalculatorTask(List<double[]> population, int start, int end,int THRESHOLD) {
            this.positions = population;
            this.start = start;
            this.end = end;
            this.THRESHOLD=THRESHOLD;
        }

        @Override
        protected List<Double> compute() {
            int length = end - start;
            if (length <= THRESHOLD) {
                return computeDirectly();
            }

            int mid = (start + end) / 2;
            FitnessCalculatorTask leftTask = new FitnessCalculatorTask(positions, start, mid,THRESHOLD);
            FitnessCalculatorTask rightTask = new FitnessCalculatorTask(positions, mid, end,THRESHOLD);

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
                double[] individual = positions.get(i);
                double fit = computeFitness(individual);
                fitness.add(fit);
            }
            return fitness;
        }
    }
    public double[] run() {
        for (int iter = 0; iter < maxIterations; iter++) {
            long time=System.currentTimeMillis();
            System.out.println("current iter: "+iter);
            for (int i = 0; i < numParticles; i++) {
                for (int j = 0; j < dimension; j++) {
                    double r1 = rand.nextDouble();
                    double r2 = rand.nextDouble();
                    velocities.get(i)[j] = w * velocities.get(i)[j] + c1 * r1 * (personalBests.get(i)[j] - positions.get(i)[j]) + c2 * r2 * (globalBest[j] - positions.get(i)[j]);
                    positions.get(i)[j] += velocities.get(i)[j];
                }
            }
            fitness=evaluate(positions);
            for (int i=0;i<fitness.size();i++){
                double fit=fitness.get(i);
                if (fit > personalBestFitness[i]) {
                    personalBestFitness[i] = fit;
                    personalBests.set(i, positions.get(i).clone());
                }
                if (fit > globalBestFitness || globalBestFitness != globalBestFitness) {
                    globalBestFitness = fit;
                    globalBest = positions.get(i).clone();
                }
            }
            System.out.println("iter "+iter+" time: "+(System.currentTimeMillis()-time));
        }
        DecimalFormat df = new DecimalFormat("0.0000000000");
        System.out.println("Global Best Value: " + df.format(globalBestFitness));
        return globalBest;
//        System.out.println("Global Best Position: " + Arrays.toString(globalBest));
//        List<double[]> centroids=decode(globalBest);
//        List<List<double[]>>result=clustering(centroids);
//        System.out.println("Best result:");
//        for (int i=0;i<result.size();i++){
//            System.out.println("Cluster "+i+":");
//            for (double[]point:result.get(i)){
//                System.out.println(Arrays.toString(point));
//            }
//        }
    }
    public List<List<double[]>> clustering(List<double[]> centroids) {
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

    List<double[]> decode(double[] particle) {
        List<double[]> centroids = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            double[] centroid = new double[dataLength];
            int offset = i * dataLength;
            for (int j = 0; j < dataLength; j++) {
                centroid[j] = particle[offset + j];
            }
            centroids.add(centroid);
        }
        return centroids;
    }

    private double distance(double[] a, double[] b) {
        double distance = 0;
        for (int i = 0; i < a.length; i++) {
            distance += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(distance);
    }

}

