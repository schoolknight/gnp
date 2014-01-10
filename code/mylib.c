
#include<math.h>

#define MAX_ERROR 0.001

double calcDist(double keyK,double keyL){
  return (keyK * keyL * keyL)/sin(keyK * keyL);


}

//clac the init point
int initCalc(double keyK,double keyMax,int keyNum,double* keyArray){
  int cur = 0;
  double curL = keyMax;
  int curP = 1,tmpL;
  int i;
  int j;

  for(i = 0;i < keyNum;i ++){
    curP *= 2;
    curL = curL / 2;
    for(j = 1;j < curP;j ++)
      if (j % 2 == 1)
        keyArray[cur ++] = calcDist(keyK,curL * j);
  }
  return 1;
}




double calcRe(double keyK,double keyMax,int keyNum,double* keyArray,double keyV){
  int curP = 0;
  double fL = 0,eL = keyMax;
  int i;
  for(i = 0;i < keyNum;i ++){
    //printf("%d  %f  %f\n",curP,fL,eL);    
    if (keyV < keyArray[curP]){
      curP = curP * 2 + 1;
      eL = (eL + fL) / 2;
    }
    else if (keyV == keyArray[curP])
      return (eL + fL) / 2;
    else{
      curP = curP * 2 + 2;
      fL = (eL + fL) / 2;
    }
  }
  double curL,curV;
  while(eL - fL > MAX_ERROR){
    //printf("%f  %f \n",fL,eL);
    curL = (eL + fL) / 2;
    curV = calcDist(keyK,curL);
    if (keyV < curV)      
      eL = (eL + fL) / 2;
    else if (keyV == keyArray[curP])
      return curL;
    else
      fL = (eL + fL) / 2;
  }
  return eL;
}
