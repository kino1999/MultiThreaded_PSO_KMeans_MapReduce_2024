package org.example;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.hdfs.protocol.DatanodeInfo;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class PSOKMeansDriver extends Configured implements Tool {
    static Configuration conf;
//    static String master="hdfs://localhost:9000/";
//   static String dataPath = "../../../data/";
//   final static String popPath = "popfiles/pop";
//   final static String outputPath = "output";

    static String master="hdfs://hadoop01:9000";
    static String dataPath = "hdfs://hadoop01:9000/";
    final static String popPath = "hdfs://hadoop01:9000/popfiles/pop";
    final static String outputPath = "hdfs://hadoop01:9000/output";

//    static String master="hdfs://pcu-75:9000";
//    static String dataPath = "hdfs://pcu-75:9000/";
//    final static String popPath="hdfs://pcu-75:9000/popfiles/pop";
//    final static String outputPath = "hdfs://pcu-75:9000/output";
    final static String outputFile = outputPath + "/part-r-00000";
    private static int k;
    private static int numParticles;
    private static List<double[]> positions;
    private static int islandSize;
    private static int islandNumParticles;
    static int maxIterations;
    private static double w;
    private static double c1;
    private static double c2;
    private static List<double[]> data;
    private static int dataLength;
    private static int dimension;
    private static Random random;
    private static List<double[]> personalBests;
    static List<Double> personalBestFitness;

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        setGlobalParams(conf,args);
        Job job = Job.getInstance(conf, "GKA");
        job.getConfiguration().setLong("mapreduce.task.timeout", 0);
        job.setJarByClass(PSOKMeansDriver.class);
        job.setMapperClass(PSOKMeansMapper.class);
        job.setReducerClass(PSOKMeansReducer.class);
        job.setInputFormatClass(LineAsSplitInputFormat.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(popPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        job.addCacheFile(new URI(dataPath));
        return job.waitForCompletion(true) ? 0 : 1;
    }
    public static void setupParams(Configuration conf) throws IOException {
        k = 50;
        numParticles = 200;
        maxIterations = 100;
        w=0.5;
        c1=1.5;
        c2=1.5;
        random = new Random(new Date().getTime());
        dataLength = data.get(0).length;
        dimension = k*dataLength;
        islandSize = getActiveNodes(conf);
//        islandSize = 1;
        islandNumParticles = numParticles /islandSize;
        numParticles = islandNumParticles *islandSize;
        System.out.println(data.size());
        System.out.println("data path: "+dataPath);
        System.out.println("island size: "+islandSize);
        System.out.println("island population: "+ islandNumParticles);
        System.out.println("total population: "+ numParticles);
        System.out.println("param nums: "+ dimension);
    }
    public static void setGlobalParams(Configuration conf,String []args) {
        conf.setInt("k", k);
        conf.setInt("paramNum", dimension);
        conf.setInt("numParticles", islandNumParticles);
        conf.setInt("dataLength", dataLength);
        conf.setInt("maxIterations", maxIterations);
        conf.setInt("islandSize", islandSize);
        conf.setDouble("w", w);
        conf.setDouble("c1", c1);
        conf.setDouble("c2", c2);
        conf.set("dataFileName", args[0]);
    }

    public static void main(String[] args) throws Exception {
        dataPath += args[0];
        data = readData(dataPath);
        System.out.println("data read finished");
        conf = new Configuration();
        FileSystem fileSystem = FileSystem.get(conf);
        Path path = new Path(outputPath);
        while (fileSystem.exists(path)) {
            fileSystem.delete(path, true);
        }
        setupParams(conf);
        long time = System.currentTimeMillis();
        positions=initPositions();
        writePop(positions, islandNumParticles);
        ToolRunner.run(new Configuration(), new PSOKMeansDriver(), args);
        readPopAndScore();
        int bestIndex = getBestIndex();
        double[] bestIndividual = personalBests.get(bestIndex);
        System.out.println("time cost " + (System.currentTimeMillis() - time) + "ms");
        List<List<double[]>> result = clustering(decode(bestIndividual));
        System.out.println("Silhouette Coefficient: " + scores.silhouetteCoefficient(result));
        System.out.println("Davies Bouldin Index: " + scores.daviesBouldinIndex(result));
        System.out.println("Calinski Harabasz: " + scores.calinskiHarabaszIndex(result));
        System.out.println("Variance Evaluation: " + scores.varianceEvaluation(result));
    }
    private static List<double[]> initPositions() {
        Random rand=new Random();
        List<double[]>positions = new ArrayList<>();
        for (int i = 0; i < numParticles; i++) {
            List<Integer> selected = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                int index = rand.nextInt(data.size());
                if (selected.contains(index)) {
                    j--;
                    continue;
                }
                selected.add(index);
            }
            double[] position = new double[dimension];
            for (int j = 0; j < k; j++) {
                double[] a = data.get(selected.get(j));
                for (int m = 0; m < dataLength; m++) {
                    position[j * dataLength + m] = a[m];
                }
            }
            positions.add(position);
        }
        return positions;
    }
    private static int getBestIndex() {
        int maxFitnessIndex = 0;
        double maxFitness = Double.MIN_VALUE;
        for (int i = 0; i < personalBestFitness.size(); i++) {
            if (personalBestFitness.get(i) > maxFitness) {
                maxFitnessIndex = i;
                maxFitness = personalBestFitness.get(i);
            }
        }
        return maxFitnessIndex;
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
    public static void readPopAndScore() throws IOException {
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(configuration);
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(outputFile)), StandardCharsets.UTF_8));
        String line;
        List<double[]> newPopulation = new ArrayList<>();
        List<Double> newFitness = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            int beginIndex = 0;
            while (line.charAt(beginIndex) != '\t') {
                beginIndex++;
            }
            String[] individualString = line.substring(beginIndex + 1).split(",");
            double[] individual = new double[dimension];
            double score = Double.parseDouble(individualString[0]);
            if (score == -1)
                continue;
            for (int i = 0, j = 1; i < dimension; i++, j++) {
                individual[i] = Double.parseDouble(individualString[j]);
            }
            newFitness.add(score);
            newPopulation.add(individual);
        }
        personalBests = newPopulation;
        personalBestFitness = newFitness;
    }
    private static void writePop(List<double[]> positions, int islandNumParticles) throws IOException {
        int islandIDCount = 0;
        FileSystem fs = FileSystem.get(conf);
        BufferedWriter ow = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(popPath), true), StandardCharsets.UTF_8));
        for (int i = 0, j = 0; i < numParticles; i++, j++) {
            double[] ind = positions.get(i);
            if (j == 0) {
                if (islandIDCount > islandSize)
                    break;
                ow.write(String.valueOf(islandIDCount));
                ow.write('\t');
            }
            for (double d : ind) {
                ow.write(String.valueOf(d));
                ow.write(",");
            }
            if (j == islandNumParticles - 1) {
                ow.write("\r\n");
                j = -1;
                islandIDCount++;
            }
        }
        ow.flush();
        ow.close();
    }
    public static List<List<double[]>> clustering(List<double[]> centroids) {
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
    static List<double[]> decode(double[] particle) {
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
    private static double distance(double[] a, double[] b) {
        double distance = 0;
        for (int i = 0; i < a.length; i++) {
            distance += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(distance);
    }
    private static int getActiveNodes(Configuration conf) throws IOException {
        conf.set("fs.defaultFS", master); // 用你的Hadoop master节点地址替换

        // 获取分布式文件系统
        DistributedFileSystem dfs = (DistributedFileSystem) FileSystem.get(conf);

        // 获取数据节点信息
        DatanodeInfo[] dataNodeStats = dfs.getDataNodeStats();

        // 打印节点数量
        System.out.println("Number of DataNodes: " + dataNodeStats.length);
        return dataNodeStats.length;
    }

}
