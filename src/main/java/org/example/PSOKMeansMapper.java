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
public class PSOKMeansMapper extends Mapper<LongWritable, Text, IntWritable,Text> {
    private static List<double[]> data = new ArrayList<>();
    private int k;
    private int numParticles;
    private int dimension;
    private int maxIterations;
    private double w;
    private double c1;
    private double c2;


    protected void setup(Context context) throws IOException {
        Configuration conf = context.getConfiguration();
        k = conf.getInt("k", 0);
        numParticles = conf.getInt("numParticles", 0);
        dimension = conf.getInt("paramNum", 0);
        maxIterations = conf.getInt("maxIterations", 0);
        w = conf.getDouble("w", 0);
        c1 = conf.getDouble("c1", 0);
        c2 = conf.getDouble("c2", 0);
        if (data.size() == 0) {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null && cacheFiles.length > 0) {
                for (URI cacheUri : cacheFiles) {
                    Path cachePath = new Path(cacheUri);
                    if (cachePath.getName().equals(conf.get("dataFileName"))) {
                        // 读取 data.csv 文件
                        FileSystem fs = FileSystem.get(cacheUri, conf);
                        data = readDataFile(fs, cachePath);
                    }
                }
            }
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        List<double[]> positions = getPositions(value);
        double[] globalBest;
        double globalBestFitness;
        FJPSOKMeans psokMeans = new FJPSOKMeans(numParticles, maxIterations, k, data, w, c1, c2,positions);
        globalBest = psokMeans.run();
        globalBestFitness = psokMeans.computeFitness(globalBest);
        System.out.println(Arrays.toString(data.get(0)));
        context.write(new IntWritable(0), new Text(globalBestFitness + "," + Arrays.toString(globalBest).replace("[", "").replace("]", "").replace(" ", "")));
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
    private List<double[]> getPositions(Text value) {
        String valueS = value.toString();
        int scoreIndex = 0;
        while (valueS.charAt(scoreIndex) != '\t') {
            scoreIndex++;
        }
        String[] strings = valueS.substring(scoreIndex + 1).split(",");
        List<double[]> positions = new ArrayList<>();
        double[] individual = new double[dimension];
        for (int i = 0, j = 0; i < strings.length; i++, j++) {
            if (j < dimension)
                individual[j] = Double.parseDouble(strings[i]);
            else {
                positions.add(individual);
                individual = new double[dimension];
                j = 0;
            }
        }
        positions.add(individual);
        return positions;
    }
}
