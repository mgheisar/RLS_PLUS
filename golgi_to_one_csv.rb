#!/usr/bin/env ruby

columns = {'Count': 1,
           'Area': 2,
           'AvgAera': 3}
column = 3
base = 'summary/'

files = %w{HR0.csv PULSE_16x0.csv HR1.csv PULSE_16x1.csv}
names = %w{HR_DMSO SR_DMSO HR_Nocodazole SR_Nocodazole}

files.each_with_index { |fname, idx|
  realness, domain = names[idx].split('_').map { |s| s.upcase }
  File.open(base + fname) { |f|
    f.readline()
    f.each_line { |line|
      a = line.split(',')
      print([realness, domain, a[column]].join("\t"), "\n")
      File.open(base + 'res', "a") { |f|
        f.write([realness, domain, a[column]].join("\t"), "\n")
      }
    }
  }
}
