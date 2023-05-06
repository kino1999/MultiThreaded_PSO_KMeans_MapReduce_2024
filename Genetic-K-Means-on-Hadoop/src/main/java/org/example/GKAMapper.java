package org.example;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;
public class GKAMapper extends Mapper<LongWritable, Text, IntWritable,Text> {
    private int islandID;
    private int islandSize;
    private static List<double[]> data = new ArrayList<>();
    private int k;
    private int paramNum;
    private int maxIterations;
    private double crossoverProbability;
    private double mutationProbability;
    private List<Double> fitness;
    private boolean isLastRound;

    protected void setup(Context context) throws IOException {
        Configuration conf = context.getConfiguration();
        k = conf.getInt("k", 0);
        paramNum = conf.getInt("paramNum", 0);
        maxIterations = conf.getInt("maxIterations", 0);
        crossoverProbability = conf.getDouble("crossoverProbability", 0);
        mutationProbability = conf.getDouble("mutationProbability", 0);
        islandSize = conf.getInt("islandSize", 0);
        if (data.size() == 0) {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                for (URI cacheUri : cacheFiles) {
                    Path cachePath = new Path(cacheUri);
                    if (cachePath.getName().equals(conf.get("dataFileName"))) {
                        // 读取 data.csv 文件
                        FileSystem fs = FileSystem.get(cacheUri, conf);
                        data=readDataFile(fs, cachePath);
                    }
                }
            }
        }
        isLastRound = conf.getBoolean("isLastRound", false);
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        List<int[]> population = getPopulationAndIslandID(value);
        int populationSize = population.size();
        ForkJoinGeneticKMeans GKA = new ForkJoinGeneticKMeans(k, population, populationSize, maxIterations, crossoverProbability, mutationProbability, data);
        population = GKA.run();
        fitness = GKA.evaluate(population);
        int[] bestIndividual = null;
        double bestFitness = Double.MIN_VALUE;
        int bestIndex = getBestIndex();
        if (fitness.get(bestIndex) > bestFitness) {
            bestIndividual = Arrays.copyOf(population.get(bestIndex), population.get(bestIndex).length);
            bestFitness = fitness.get(bestIndex);
        }
        if (bestIndividual==null){
            bestIndividual=population.get(0);
        }
        System.out.println(fitness);
        if (isLastRound) {
            context.write(new IntWritable(0), new Text(bestFitness + "," + Arrays.toString(bestIndividual).replace("[", "").replace("]", "").replace(" ", "")));
        } else {
            migrateBestIndividual(bestIndividual, context);
            IntWritable intWritableIslandID = new IntWritable(islandID);
            selectSurvivor(population, fitness, context, intWritableIslandID);
        }
    }
    private void selectSurvivor(List<int[]> population, List<Double> fitness,Context context,IntWritable intWritableIslandID) throws IOException, InterruptedException {
        int count=0;
        int survivorNum=population.size()-islandSize+1;
        double sum;
        double[] cumulativeFitness = new double[fitness.size()];
        cumulativeFitness[0] = fitness.get(0);
        for (int i = 1; i < fitness.size(); i++) {
            cumulativeFitness[i] = cumulativeFitness[i - 1] + fitness.get(i);
        }
        sum = cumulativeFitness[cumulativeFitness.length - 1];

        while (count<survivorNum){
            double r = Math.random() * sum;
            for (int j = 0; j < cumulativeFitness.length; j++) {
                if (r <= cumulativeFitness[j]) {
                    context.write(intWritableIslandID, new Text(Arrays.toString(population.get(j)).replace("[", "").replace("]", "").replace(" ", "")));
                    count++;
                    break;
                }
            }
        }
    }
    private void migrateBestIndividual(int[] bestIndividual, Context context) throws IOException, InterruptedException {
        Text value = new Text(Arrays.toString(bestIndividual).replace("[", "").replace("]", "").replace(" ", ""));
        for (int i = 0; i < islandSize; i++) {
            if (i == islandID)
                continue;
            context.write(new IntWritable(i), value);
        }
    }

    private List<int[]> getPopulationAndIslandID(Text value) {
        String valueS = value.toString();
        StringBuilder idString = new StringBuilder();
        int scoreIndex = 0;
        while (valueS.charAt(scoreIndex) != '\t') {
            idString.append(valueS.charAt(scoreIndex));
            scoreIndex++;
        }
        islandID = Integer.parseInt(idString.toString());
        String[] strings = valueS.substring(scoreIndex + 1).split(",");
        List<int[]> population = new ArrayList<>();
        int[] individual = new int[paramNum];
        for (int i = 0, j = 0; i < strings.length; i++, j++) {
            if (j < paramNum)
                individual[j] = Integer.parseInt(strings[i]);
            else {
                population.add(individual);
                individual = new int[paramNum];
                j = 0;
            }
        }
        population.add(individual);
        return population;
    }

    private int getBestIndex() {
        int maxFitnessIndex = 0;
        double maxFitness = Double.MIN_VALUE;
        for (int i = 0; i < fitness.size(); i++) {
            if (fitness.get(i) > maxFitness) {
                maxFitnessIndex = i;
                maxFitness = fitness.get(i);
            }
        }
        return maxFitnessIndex;
    }
    public static List<double[]> readDataFile(FileSystem fs,Path path) throws IOException {
        List<double[]> data = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path), StandardCharsets.UTF_8));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            String[] s = line.split(",");
            data.add(new double[s.length]);
            for (int i = 0; i < s.length; i++) {
                data.get(index)[i] = Double.parseDouble(s[i]);
            }
            data.set(index, data.get(index));
            index++;
        }
        return data;
    }
}
