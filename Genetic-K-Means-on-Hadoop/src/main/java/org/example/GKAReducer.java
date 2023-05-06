package org.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;

public class GKAReducer extends Reducer<IntWritable, Text,IntWritable, Text> {
    private boolean isLastRound;

    protected void setup(Context context) {
        Configuration conf = context.getConfiguration();
        isLastRound = conf.getBoolean("isLastRound", false);
    }

    @Override
    protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        if (isLastRound) {
            for (Text text : values) {
                context.write(key, text);
            }
        } else {
            StringBuilder valueOut = new StringBuilder();
            int count=0;
            for (Text text : values) {
                valueOut.append(text).append(",");
                count++;
            }
            valueOut.delete(valueOut.length() - 1, valueOut.length());
            context.write(key, new Text(valueOut.toString()));
        }
    }
}
