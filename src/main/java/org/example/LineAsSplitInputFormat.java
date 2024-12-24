package org.example;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;
import org.apache.hadoop.util.LineReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LineAsSplitInputFormat extends FileInputFormat<LongWritable, Text> {

    @Override
    public List<InputSplit> getSplits(JobContext job) throws IOException {
        List<InputSplit> splits = new ArrayList<>();
        List<FileStatus> files = listStatus(job);
        for (FileStatus file : files) {
            Path path = file.getPath();
            FileSystem fs = path.getFileSystem(job.getConfiguration());
            FSDataInputStream inputStream = null;
            try {
                inputStream = fs.open(path);
                LineReader lineReader = new LineReader(inputStream, job.getConfiguration());
                long offset = 0;
                long length = file.getLen();
                long lineOffset = 0;
                Text line = new Text();
                while (offset < length) {
                    int newSize = lineReader.readLine(line);
                    if (newSize == 0) {
                        break;
                    }
                    splits.add(new FileSplit(path, lineOffset, newSize, null));
                    offset += newSize;
                    lineOffset = offset;
                }
            } finally {
                if (inputStream != null) {
                    inputStream.close();
                }
            }
        }
        return splits;
    }

    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split,
                                                               TaskAttemptContext context) throws IOException, InterruptedException {
        LineAsSplitRecordReader reader = new LineAsSplitRecordReader();
        reader.initialize(split, context);
        return reader;
    }

    public static class LineAsSplitRecordReader extends RecordReader<LongWritable, Text> {

        private LineRecordReader lineRecordReader;

        public LineAsSplitRecordReader() {
            this.lineRecordReader = new LineRecordReader();
        }

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
            this.lineRecordReader.initialize(split, context);
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            return this.lineRecordReader.nextKeyValue();
        }

        @Override
        public LongWritable getCurrentKey() throws IOException, InterruptedException {
            return this.lineRecordReader.getCurrentKey();
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException {
            return this.lineRecordReader.getCurrentValue();
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return this.lineRecordReader.getProgress();
        }

        @Override
        public void close() throws IOException {
            this.lineRecordReader.close();
        }
    }
}

