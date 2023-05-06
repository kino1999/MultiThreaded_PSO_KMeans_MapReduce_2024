package org.example;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.filecache.DistributedCache;

import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class GKADriver extends Configured implements Tool {
    static Configuration conf;
//   static String dataPath = "../datasets/";
//   final static String popPath = "popfiles/pop";
//   final static String outputPath = "output";

    static String dataPath = "hdfs://hadoop01:9000/";
    final static String popPath = "hdfs://hadoop01:9000/popfiles/pop";
    final static String outputPath = "hdfs://hadoop01:9000/output";

//    static String dataPath = "hdfs://pcu-75:9000/";
//    final static String popPath="hdfs://pcu-75:9000/popfiles/pop";
//    final static String outputPath = "hdfs://pcu-75:9000/output";


    final static String outputFile = outputPath + "/part-r-00000";
    private static int k;
    private static int populationSize;
    private static int islandSize;
    private static int islandPopulationSize;
    private static int communicationCases;
    static int maxIterations;
    static double crossoverProbability;
    static double mutationProbability;
    private static List<double[]> data;
    private static int dataLength;
    private static int paramNum;
    private static Random random;
    private static List<int[]> population;
    static List<Double> fitness;
    static boolean isLastRound = false;

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        setGlobalParams(conf,args);
        if (isLastRound)
            conf.setBoolean("isLastRound", true);
        Job job = Job.getInstance(conf, "GKA");
        job.getConfiguration().setLong("mapreduce.task.timeout", 0);
        job.setJarByClass(GKADriver.class);
        job.setMapperClass(GKAMapper.class);
        job.setReducerClass(GKAReducer.class);
        job.setInputFormatClass(LineAsSplitInputFormat.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(popPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        job.addCacheFile(new URI(dataPath));
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void setupParams(){
        k = 50;
        populationSize = 400;
        maxIterations = 30;
        crossoverProbability = 0.6;
        mutationProbability = 0.05;
        communicationCases = 3;
        random = new Random(new Date().getTime());
        dataLength = data.get(0).length;
        paramNum = data.size();
        islandSize = 20;
        islandPopulationSize = populationSize/islandSize;
        populationSize=islandPopulationSize*islandSize;
        System.out.println(data.size());
        System.out.println("data path: "+dataPath);
        System.out.println("island size: "+islandSize);
        System.out.println("island population: "+islandPopulationSize);
        System.out.println("total population: "+populationSize);
        System.out.println("param nums: "+paramNum);
    }
    public static void setGlobalParams(Configuration conf,String []args) {
        conf.setInt("k", k);
        conf.setInt("paramNum", paramNum);
        conf.setInt("dataLength", dataLength);
        conf.setInt("maxIterations", maxIterations / communicationCases);
        conf.setInt("islandSize", islandSize);
        conf.setDouble("crossoverProbability", crossoverProbability);
        conf.setDouble("mutationProbability", mutationProbability);
        conf.set("dataFileName", args[0]);
    }

    public static void main(String[] args) throws Exception {
        dataPath+=args[0];
        data = readData(dataPath);
        System.out.println("data read finished");
        conf = new Configuration();
        FileSystem fileSystem = FileSystem.get(conf);
        Path path = new Path(outputPath);
        while (fileSystem.exists(path)) {
            fileSystem.delete(path, true);
        }
        setupParams();
        long time = System.currentTimeMillis();
        population = initPopulation();
        writePop(population, islandPopulationSize);
        for (int i = 0; i < communicationCases; i++) {
            while (fileSystem.exists(path)) {
                fileSystem.delete(path, true);
            }
            if (i == communicationCases - 1)
                isLastRound = true;
            ToolRunner.run(new Configuration(), new GKADriver(), args);
            if (!isLastRound) {
                FileUtil.copy(fileSystem, new Path(outputFile), fileSystem, new Path(popPath), false, conf);
            }
        }
        readPopAndScore();
        int bestIndex = getBestIndex();
        System.out.println(fitness.get(bestIndex));
        System.out.println(System.currentTimeMillis() - time + "ms");
    }

    private static int getBestIndex() {
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

    private static List<int[]> initPopulation() {
        List<int[]> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            int kCount = 0;
            int[] individual = new int[paramNum];
            while (kCount< k) {
                int index = random.nextInt(paramNum);
                if (individual[index]==0) {
                    individual[index] = 1;
                    kCount++;
                }
            }
            population.add(individual);
        }
        return population;
    }

    private static void writePop(List<int[]> population, int islandPopulationSize) throws IOException {
        int islandIDCount = 0;
        FileSystem fs = FileSystem.get(conf);
        BufferedWriter ow = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(popPath), true), StandardCharsets.UTF_8));
        for (int i = 0, j = 0; i < populationSize; i++, j++) {
            int[] ind = population.get(i);
            if (j == 0) {
                if (islandIDCount > islandSize)
                    break;
                ow.write(String.valueOf(islandIDCount));
                ow.write('\t');
            }
            for (int d : ind) {
                ow.write(String.valueOf(d));
                ow.write(",");
            }
            if (j == islandPopulationSize - 1) {
                ow.write("\r\n");
                j = -1;
                islandIDCount++;
            }
        }
        ow.flush();
        ow.close();
    }

    public static double[] maxMinNormalize(double[] doubles) {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        double[] newArr = new double[doubles.length];
        for (double d : doubles) {
            if (max < d)
                max = d;
            if (min > d)
                min = d;
        }
        for (int i = 0; i < doubles.length; i++) {
            newArr[i] = (doubles[i] - min) / (max - min);
        }
        return newArr;
    }

    public static List<double[]> preProcess(List<double[]> preData) {
        List<double[]> data = new ArrayList<>();
        data.add(preData.get(0));
        for (int i = 1; i < preData.size(); i++) {
            for (int j = 0; j < data.size(); j++) {
                if (Arrays.equals(preData.get(i), data.get(j))) {
                    break;
                } else if (j == data.size() - 1) {
                    data.add(preData.get(i));
                }
            }
        }
        return data;
    }

    public static List<double[]> readData(String path) throws IOException {
        List<double[]> data = new ArrayList<>();
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(configuration);
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(path)), StandardCharsets.UTF_8));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            String[] s = line.split(",");
            data.add(new double[s.length]);
            for (int i = 0; i < s.length; i++) {
                data.get(index)[i] = Double.parseDouble(s[i]);
            }
            data.set(index, maxMinNormalize(data.get(index)));
            index++;
        }
        return data;
    }

    public static void readPopAndScore() throws IOException {
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(configuration);
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(outputFile)), StandardCharsets.UTF_8));
        String line;
        List<int[]> newPopulation = new ArrayList<>();
        List<Double> newFitness = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            int beginIndex = 0;
            while (line.charAt(beginIndex) != '\t') {
                beginIndex++;
            }
            String[] individualString = line.substring(beginIndex + 1).split(",");
            int[] individual = new int[paramNum];
            double score = Double.parseDouble(individualString[0]);
            if (score == -1)
                continue;
            for (int i = 0, j = 1; i < paramNum; i++, j++) {
                individual[i] = Integer.parseInt(individualString[j]);
            }
            newFitness.add(score);
            newPopulation.add(individual);
        }
        population = newPopulation;
        fitness = newFitness;
    }
}
