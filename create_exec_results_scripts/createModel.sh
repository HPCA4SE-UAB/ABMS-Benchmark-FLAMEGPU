#!/bin/bash
echo "Hit \"d\" to delete *.cu files, other key only make model: "
read -n 1 key
if [ $key == "d" ];
then
   rm FLAMEGPU_kernals.cu
   rm functions.cu
   rm header.h
   rm io.cu
   rm main.cu
   rm simulation.cu
   rm visualisation.cu
   rm ./model
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/FLAMEGPU_kernals.xslt -OUT FLAMEGPU_kernals.cu
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/functions.xslt -OUT functions.cu
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/header.xslt -OUT header.h
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/io.xslt -OUT io.cu
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/main.xslt -OUT main.cu
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/simulation.xslt -OUT simulation.cu
   java -jar ~/FLAMEGPU/xalan/xalan-j_2_7_2/xalan.jar -IN persons_interact.xml -XSL xslt/visualisation.xslt -OUT visualisation.cu
   make
else
   rm ./model
   make
fi

