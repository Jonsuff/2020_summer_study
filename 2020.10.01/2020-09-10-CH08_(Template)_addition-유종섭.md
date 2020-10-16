---
layout: post
title:  "CH08 Template"
date:   2020-09-10 14:09:13
categories: C++
---



# CH08 템플릿(Template)

- 템플릿(Template) 이란?

  1. 함수나 클래스를 개별적으로 다시 작성하지 않아도, 여러 자료형으로 사용할 수 있도록 만들어 놓은 툴.
  2. 템플릿의 종류로는 (1) 클래스 템플릿(class template), (2) 함수 템플릿(function template), (3) 타입 템플릿(type template), (4) 변수 템플릿(Variable Template)이 있다.

- 함수 템플릿(function template)으로 보는 템플릿의 의미: 

  > 함수를 만들때 함수의 기능은 명확하지만, 자료형을 모호하게 두는 것.

  함수 오버로딩의 경우, 함수의 기능이 동일함에도 불구하고 다음과 같이 데이터와 데이터 타입에 따라 무수히 많은 함수를 일일히 정의해야 하는 경우가 있다.

  ```cpp
  // a와 b의 조건이 참일때 a를 반환하고, 거짓일때 b를 반환한다.
  // int 타입의 함수
  int min(int a, int b) {
      return (a < b) ? a : b;
  }
  
  // long 타입의 함수
  long min(long a, long b) {
      return (a < b) ? a : b;
  }
  
  // char 타입의 함수
  char min(char a, char b) {
      return (a < b) ? a : b;
  }
  ```

  이처럼 함수의 기능은 동일하지만 매개변수의 데이터 타입에 따라 다르게 함수를 제작해야 하는 불편함이 있다.

  만약 다음과 같이 함수 템플릿을 사용하면 변수의 자료형을 모호하게 지정할 수 있고, 컴파일 시점에 데이터 타입이 확정되는 기능을 이용할 수 있기때문에 보다 단순한 코딩을 할 수 있다.

  ```cpp
  template<typename T>
  T min(T a, T b) {
      return (a < b) ? a : b;
  }
  ```

  예시 : 

  ```cpp
  #include <iostream>
  #include <typeinfo>
  
  int min_func(int a, int b) {
      return (a > b) ? a : b;
  }
  
  template <typename T>
  T min_temp(T a, T b) {
      return (a < b) ? a : b;
  }
  
  int main()
  {
      std::cout <<"using function : " << min_func(10, 20) << std::endl;
      std::cout << "using template with double data : " << min_temp(5.5, 15.9) << std::endl;
      std::cout << "type of result2 : " << typeid(min_temp(5.5, 15.9)).name() << std::endl;
      std::cout << "using template with int data : " << min_temp(10, 20) << std::endl;
      std::cout << "type of result3 : " << typeid(min_temp(10, 20)).name() << std::endl;
      return 0;
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex1.png)





## 8_01 템플릿의 포맷

- 템플릿의 선언 예시는 다음과 같다.

  ```cpp
  template<typename T>
  void f(T s){
      std::cout << s << std::endl;
  }
  ```

  > template <템플릿 매개변수-리스트>
  >
  > 함수 또는 클래스 선언문

  - 템플릿 매개변수-리스트는 하나 이상의 매개변수를 사용할 수 있다(다수 사용한다면 콤마로 구별한다).

  - 템플릿 매개변수는 다음과 같이 세 종류로 나누어진다.

    1. 타입 템플릿 매개변수 : typename / class 지시어 사용

       > 함수나 클래스 내부에서 사용하는 데이터 타입을 가리키는 매개변수
       >
       > template<typename T\>

    2. 템플릿 템플릿 매개변수 : template 안에 또다른 템플릿 사용

       > 또다른 클래스 템플릿을 가리키는 매개변수
       >
       > template<typename T, template<typename ELEM\> class CONT = std::deque>

    3. 타입이 아닌 템플릿 매개변수 : 초기화 하고자 하는 지역변수의 자료형 사용

       > 배열 크기를 지정하거나, 클래스나 함수의 지역 변수를 초기화 시키는 목적
       >
       > template<int N\>

- C++ 언어에서 제공하는 템플릿의 종류는 다음과 같다.

  1. 클래스 템플릿(class template) : 

     클래스 내부에서 사용하는 멤버 변수 또는 멤버 함수의 데이터 타입을 지정할 때 사용한다.

     ```cpp
     // 클래스 예제
     class Data {
         int data;
     public:
         Data(int d){
             data = d;
         }
         void SetData(int d){
             data = d;
         }
         int GetData(){
             return data;
         }
     };
     
     // 위의 클래스 예제를 템플릿화
     template<typename T>
     class Data {
         T data;
     public:
         Data(T d){
             data = d;
         }
         void SetData(T d){
             data = d;
         }
         T GetData(){
             return data;
         }
     };
     ```

     이런식으로 선언한 클래스 템플릿은 main에서 사용할때 기존 클래스처럼 Data d1(10);으로 사용하면 문제가 발생한다. 그 이유는 다음과 같은 측면에서 살펴볼 수 있다.

     - 객체 생성의 순서 : 

       객체 생성의 순서는 메모리 할당 -> 생성자 호출 .... 이러한 순서로 진행된다. 만약 기존처럼 Data d1(10);로 사용하는 경우에는 d1이라는 이름으로 메모리 공간이 할당되어야 한다.

       하지만 템플릿은 객체의 자료형을 모호하게 만들었기 때문에 현재 T가 어떤 데이터형을 사용해야 할지 결정이 나지 않은 상태이다. 

       T의 자료형이 결정나는 시점은 생성자가 호출될때(괄호안의 10이 호출될 때) T가 int형 데이터인지 알 수 있으므로 이전과 같은 방법을 사용하면 메모리 할당을 할 수 없는 것이다.

     따라서 클래스 템플릿에서 Data 객체를 만들기 위해서는 구체적으로 어떤 타입으로 템플릿을 구체화 시킬지 명시적으로 선언해야 한다.

     ```cpp
     int main(void) {
         Data<int> d1(0);
         // Data d1(10); ---> error!
         std::cout << "default d1 data : " << d1.GetData() << std::endl;
         
         d1.SetData(10);
         
         Data<char> d2('a');
         
         std::cout << "set d1 data : " << d1.GetData() << std::endl;
         std::cout << "set d2 data : " << d2.GetData() << std::endl;
     }
     ```

     결과 : 

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex2.png)

  2. 함수 템플릿(function template) : 

     앞선 예시로 대체한다.

  3. 타입 템플릿(alias template) : 

     using 또는 typedef 선언문에서 사용하는 데이터 타입을 지정할 때 사용한다.

     ```cpp
     // 템플릿을 포인터로 사용한다.
     template<class T>
     using ptr = T*;
     
     ptr<int> x; // int *x의 선언과 같다.
     ```

  4. 변수 템플릿(variable template) : 

     C++14 이후부터 변수에 적용하는 템플릿을 다음과 같이 사용할 수 있다.

     ```cpp
     template<typename T>
     constexpr T pi = T(3.1415926535897932);
     ```

     변수 템플릿은 중요하지 않다(많이 사용하지 않는다).

- 템플릿은 메타프로그래밍이다 : 

  메타프로그래밍(Metaprogramming)이란 프로그램을 일종의 데이터로 취급하고 컴파일 시점에 실행 가능한 프로그램으로 변경시켜주는 프로그래밍 기법이다.

  다시 말해 템플릿은 클래스나 함수를 생성하기 위해 선언되는 것이지 실제 함수나 클래스를 의미하지 않는다. 즉 템플릿은 main()에서 자료형을 확정지어 선언될 때 비로소 사용 가능한 클래스나 함수로 변경된다.

  ```cpp
  template<typename T> class Vector // 실제 클래스가 아니다.
      
  int main(void){
      Vector<int> vec; // 비로소 int 자료형으로 사용 가능한 클래스가 된다.
  }
  ```

  위와같이 클래스 템플릿으로 클래스를 생성하는 과정을 **클래스 템플릿의 인스턴스화(class template instantiation)**라고 한다. 그리고 생성된 클래스를 **클래스 템플릿의 인스턴스**라 한다.

- 클래스 테플릿의 인스턴스화 과정 : 

  **클래스 템플릿** -> **클래스 템플릿 변환** -> **클래스 템플릿의 인스턴스**

- 클래스 템플릿의 암시적/명시적 선언 방법 : 

  ```cpp
  #include <iostream>
  #include <typeinfo>
  
  template <typename T>
  T min_temp(T a, T b) {
      return (a < b) ? a : b;
  }
  
  int main(void){
      // 암시적인 선언 방법
  	int result = min_temp(3, 10);
      std::cout << "result : " << result << std::endl;
      
      //명시적인 선언 방법
      std::cout << "result : " << min_temp<double>(3.5, 10.8) << std::endl;
      
      // 타입 확인
      std::cout << "암시적 : " << typeid(result).name() << std::endl;
      std::cout << "명시적 : " << typeid(min_temp<double>(3.5, 10.8)).name() << std::endl;
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex3.png)

  주의할 사항은 위의 예시에서 min_temp\<int>(3, 6.8)과 같이 자료형을 섞어서 사용했다면 <>내부의 자료형이 우선순위가 높기 때문에 int로 선언된다는 점이다.





## 8_02 템플릿 매개변수

- 타입 템플릿 매개변수 :

  가장 많이 사용하는 템플릿 매개변수이고, 함수나 클래스에 특정 데이터 타입을 명시하는 목적으로 사용한다.

  - template<typename T\>
  - template<class C\>

  이때 typename과 class 뒤에 나오는 T, C를 매개변수(또는 식별자)라고 부르는데, 이는 함수 또는 클래스 내부에서 사용되는 데이터 타입을 나타낸다.

  

  템플릿 변환시에 템플릿 인수로 사용할 수 있는 데이터 타입은 다음과 같다.

  - int 타입이나 char같은 기본 타입

    > ```cpp
    > int result = min_temp(3, 10);
    > ```

  - 클래스와 구조체 그리고 공유체와 같은 묶음 타입

    > ```cpp
    > Data<int> d1(0);
    > ```

  - T::value_type처럼 T라고 하는 묶음 타입

    > ```cpp
    > std::map<int, const char *>::value_type
    > // 위는 std::map<int, const char *>클래스 내부에서 선언된 데이터 타입
    > // typedef std::pair<int, const char *>value_type을 가리킨다.
    > ```

  

  만약 '=' 대입 연산자 다음에 나오는 데이터 타입이 있다면, 그것은 템플릿 변환 시 템플릿 인수로 데이터 타입을 명시하지 않았을 경우 사용되는 **디폴트 데이터 타입**이다.

  ```cpp
  #include<iostream>
  #include <typeinfo>
  
  template<class T1, class T2 = T1> // T2에 대한 자료형이 입력되지 않는다면 T1과 같은 값이 디폴트값으로 설정
  class Sum {
      T1 n1;
      T2 n2;
  public:
      Sum(T1 num1, T2 num2) {
          n1 = num1 + num2;
      }
      
      void SetData(T1 num1, T2 num2) {
          n1 = num1 + num2;
      }
      
      T1 GetData() {
          std::cout << "T1 type : " << typeid(n1).name() << std::endl;
          std::cout << "T2 type : " << typeid(n2).name() << std::endl;
          return n1;
      }
  };
  
  int main() {
      Sum<int> sum1(5, 10);
      std::cout << "sum1 assigned : " << sum1.GetData() << std::endl;
      sum1.SetData(15, 30);
      std::cout << "after setting : " << sum1.GetData() << std::endl;
      
      Sum<float> sum2(1.5, 0.89);
      std::cout << "sum2 assigned : " << sum2.GetData() << std::endl;
      sum2.SetData(15, 30);
      std::cout << "after setting : " << sum2.GetData() << std::endl;
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex6.png)

  

  `※참고`

  typename대신 typename..., 혹은 class대신 class...이 사용되는 경우는 **가변 템플릿 매개변수**를 나타내는 것이다.

  예시 : 

  ```cpp
  #include <iostream>
  
  template<typename T>
  void print(const T& t) { // 템플릿 매개변수를 참조로 사용
      std::cout << t << ", 일반 템플릿 매개변수" <<std::endl;
      
  }
  
  template<typename First, typename... Rest>
  void print(const First& first, const Rest&... rest) {
      std::cout << first << ", 가변 매개변수로 추가 ";
      print(rest...);
  }
  
  int main() {
      print(1);
      print(10, 20);
      print(100, 200, 300);
      print("first", 2, "third", 3.14159);
  }
  ```
  
결과 : 
  
![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex7.png)
  
여기서 사용된 '...'을 **줄임표**라고 부르고, 이 기호는 같은 함수를 반복적으로 호출함으로써 패턴을 재생산 하는 기능을 제공한다. 이를 **함수 인수-팩 확장(function parameter pack expansion)**이라고 한다.
  
예시2:
  
```cpp
  #include <iostream>
  
  template<typename T>
  void print(const T& t) { // 템플릿 매개변수를 참조로 사용
      std::cout << t << ", 일반 템플릿 매개변수" <<std::endl;
  }
  
  template<typename First, typename... Rest>
  void print(const First& first, const Rest&... rest) {
      std::cout << first << ", 가변 매개변수로 추가 ";
      print(rest...);
  }
  
  template<typename... Ts>
  void func(Ts... args) {
      // sizeof... 는 가변 인수의 개수를 변환한다.
      const int size = sizeof...(args) + 2;
      
      //가변 인수를 배열로 만든다.
      int res[size] = {1, args..., 2};
      std::cout << "size of " << size << std::endl;
      
      // for문 내부에 사용
      for (auto i : res) {std::cout << i << " ";}
      std::cout << std::endl;
      
      // 람다 내 캡쳐절, 또는 인수로 만들어 사용
      auto lm = [&, args...]{return print(args...);};
      lm();
  }
  
  int main() {
      func(10);
      func(48,15,86);
  }
  ```
  
결과 : 
  
![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex8.png)
  

  
`※참고`
  
클래스 템플릿과 내부 멤버 함수 템플릿은 다음과 같이 사용한다.
  
예시 : 
  
```cpp
  #include <iostream>
  
  //선언문
  template<typename T>
  struct MyClass {
      void f(int); // 1. 일반 멤버 함수
      
      template<typename T2>
      void f(T2); // 2. 멤버 함수 템플릿
  };
  
  // 1. 함수의 정의문
  template<typename T>
  void MyClass<T>::f(int x) {
      std::cout << typeid(T).name() << ", 내부 : int " << std::endl;
      std::cout << "func1 x : " << x << std::endl;
  }
  
  // 2. 멤버 함수의 정의문 - 두 개의 서로 다른 타입 템플릿 매개변수 사용가능
  template<typename T>
  template<typename T2>
  void MyClass<T>::f(T2 x) {
      std::cout << typeid(T).name() << ", 내부 " << typeid(T2).name() << std::endl;
      std::cout << "func2 x : " << x << std::endl;
  }
  
  int main() {
      MyClass<std::string> m;
      m.f('t'); // 자료형이 int가 아니므로 2. 함수에 대해 템플릿 인수 지정(암시적)
      m.f(100); // 1. 함수의 template<typename T> void MyClass<T>::f(int x) 호출
      
      MyClass<int> n;
      n.f('t'); // 자료형이 int가 아니므로 2. 함수에 대해 템플릿 인수 지정(암시적)
      n.f(100); // 1. 함수의 template<typename T> void MyClass<T>::f(int x) 호출
      n.f<int>(130); // 2. 함수의 T2에 명시적으로 int가 선언되고 x에 130이 입력
      return 0;
  }
  ```
  
결과 : 
  
![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex9.png)

​	

- 템플릿 템플릿 매개변수 : 

  아래 예시와 같이 또다른 클래스 템플릿을 가리키는 템플릿 매개변수를 말한다.

  ```cpp
  #include <iostream>
  #include <vector> // vector는 배열 기반의 구조. 새로운 원소가 추가될 때 전체 크기의 메모리 재할당 후 이전 원소를 복사하여 사용. -> 데이터 삽입시 성능 저하
  #include <deque> // deque는 vector의 단점을 보완하기 위해 등장. 새로운 원소가 추가될 때 메모리를 블록 단위로 인지하고 일부분을 메모리 재할당하여 사용. -> 성능 향상
  #include <typeinfo>
  
  template<typename T, template<typename ELEM> class CONT = std::deque>
  class Stack {
  private:
      CONT<T> elems; // 템플릿 변환을 사용하여 실제 사용하는 타입 명시
      
  public:
      void push(T const& a) {
          int vector_size = elems.size();
          // for(int i = 0; i < vector_size; i++) {
              // std::cout << i << "-th idx before push_back : " << elems[i] << std::endl;
          // }
          elems.push_back(a); // 배열의 마지막에 원소(a) 추가
          std::cout << a << " push_back finished!" << std::endl;
          // for(int i = 0; i < vector_size; i++) {
          //     std::cout << i << "-th idx after push_back : " << elems[i] << "\n" << std::endl;
          // }
      }
      void pop() {
          std::cout << "" << std::endl;
          int vector_size = elems.size();
          for(int i = 0; i < vector_size; i++) {
              std::cout << i << "-th idx before pop_back : " << elems[i] << std::endl;
          }
          std::cout << "" << std::endl;
          elems.pop_back(); // 마지막 원소 삭제
          int vector_newsize = elems.size();
          std::cout << "pop_back finished!" << std::endl;
          for(int i = 0; i < vector_newsize; i++) {
              std::cout << i << "-th idx after pop_back : " << elems[i] << ""<< std::endl;
          }
      }
      const T& top() const {
        return elems.back(); // 마지막 원소의 참조를 반환
      }
    bool empty() const {
          return elems.empty(); // 비어있는지 확인
    }
  };

  int main() {
      Stack<int, std::vector> vStack2; // T의 자료형은 int, ELEM의 자료형은 vector
      vStack2.push(200); // vector 컨테이너 elems에 200 추가, 현재 elens : [200]
      vStack2.push(130); // vector 컨테이너 elems에 130 추가, 현재 elems : [200, 130]
      
      while(vStack2.empty() == false) {
          vStack2.pop(); // vector 컨테이너 elems의 마지막 원소 삭제 : 반복문 진행 전 : [200, 130]
                          //									                진행 후 : [200]
          std::cout << "" << std::endl;
          std::cout << "vStack2 type : " << typeid(vStack2).name() << std::endl; // type 프린트
      }
  }
  ```
  
  결과 : 
  
  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex4.png)
  
  
  
- 타입이 아닌 템플릿 매개변수 :

  다음과 같이 구체적으로 명시한 데이터 타입과 그에 대응되는 인수를 템플릿 매개변수로 사용할 수 있는데, 이러한 매개변수를 타입이 아닌 템플릿 매개변수라고 부른다.

  예시 1 : 

  ```cpp
  #include<iostream>
  
  template<typename T, int rate>
  class Circle {
      T r;
  public:
      Circle(T r) :r(r+rate) {}
      T getR() {
          return r;
      }
      float getArea() {
          return r*r*3.14f;
      }
  };
  
  int main(void) {
      Circle<float, 5> c1(5.5); //템플릿 자료형을 float형태로, rate = 5, r = 5.5
      Circle<int, 3> c2(5); //템플릿 자료형을 int형태로, rate = 3, r = 5
      
      std::cout << "c1 area : " << c1.getArea() << std::endl;
      std::cout << "c2 area : " << c2.getArea() << std::endl;
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex5.png)

  

  예시 2 :

  ```cpp
  #include <iostream>
  #include <cstring>
  
  template<int i>
  class NonType {
      int array[i];         // 배열의 크기를 템플릿 인수로 제공
      const int length = i; // 배열의 크기를 상수로 사용
  
  public:
      NonType() { memset(array, 0, i * sizeof(int)); }
      
      void print() {
          // 템플릿 인수는 상수처럼 사용 가능
          std::cout << "size of " << i << std::endl;
          for (auto l: array) { std::cout << l << " "; }
          std::cout << std::endl;
      }
  };
  
  int main() {
      NonType<10> a;
      a.print();
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex10.png)

  `※참고`

  cstring 헤더를 사용하면 memset()을 이용하여 배열의 메모리를 초기화 할 수 있다.

  ```cpp
  #include <cstring>
  
  void* memset(void* ptr, int value, size_t num);
  ```

  mumset()은 위에서 정의된것 처럼, ptr로 시작하는 메모리 주소부터 num개의 바이트를 value의 값으로 채우는 작업을 수행하는 것이다. 

  주의해야 할 사항은 value값이 unsigned char로 자동 변환되는 것인데, 이때문에 생겨나는 대표적인 문제점은 다음과 같다.

  - value값은 unsigned char 형태가 아니라면 0 이외의 값이 들어가면 안된다 : 

    unsigned char는 메모리가 1byte이다. 만약 value에 정수 1이 들어간다면 정수는 4byte이기 때문에 byte단위로 처리가 되면 n = [00000001000000010000000100000001] 값으로 처리가 되므로 원하지 않는 값이 초기화가 된다. 따라서 value에는 unsigned char 형태가 아니라면 0값을 넣어주어야 한다.

  

  타입이 아닌 템플릿 매개변수로 사용가능한 데이터 타입은 다음과 같다.

  - std::nullptr_t

    > nullptr의 자료형

  - 리터럴

    > literal : 코드에 직접 삽입된 값
    >
    > 예시 : 
    >
    > ```cpp
    > bool mybool = true;
    > int x = 5;
    > std::cout << 2*3 << std::endl;
    > ```
    >
    > 위와 같이 직접 데이터값이 삽입된 것

  - const 상수

    > 한번 선언되고나면 바뀔 수 없는 상수

  - 포인터

  - lvalue 참조

  - 열거형 타입

    > C++에서 기본적으로 제공하는 자료형 이외에 사용자가 직접 만들어 사용하는 자료형
    >
    > 사용되는 키워드 : enum
    >
    > 예시 : 
    >
    > ```cpp
    > #include <iostream>
    > enum {
    >     JOB_WARRIOR = 1, // 열거형 맨 앞 멤버에 값을 지정하면 이후 멤버들은 1씩 증가
    >     JOB_MAGICIAN,
    >     JOB_ARCHER
    > };
    > 
    > int main() {
    >     int nJobType = 0;
    >     std::cout << "직업 선택. 1)전사 2)마법사 3)궁사" << std::endl;
    >     std::cin >> nJobType;
    >     
    >     switch(nJobType) {
    >         case JOB_WARRIOR: std::cout << "전사 선택완료" << std::endl; break;
    >         case JOB_MAGICIAN: std::cout << "마법사 선택완료" << std::endl; break;
    >         case JOB_ARCHER: std::cout << "궁사 선택완료" << std::endl; break;
    >     }
    >     return 0;
    > }
    > ```
    >
    > 결과 : 
    >
    > ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex11.png)





## 8_03 템플릿 특수화

- 템플릿 특수화란?

  클래스 템플릿이나 함수 템플릿을 정의할때 간혹 템플릿 인수로 적용할 수 없는 데이터 타입들이 존재한다.

  (ex) 사칙연산 수행을 위한 함수 탬플릿에는 정수와 실수만 허용하고 문자나 일반 클래스는 제외해야 한다.

  예시 : 

  ```cpp
  #include <iostream>
  #include <string>
  
  template<typename T>  
  class Add {  
  public:  
      T Sum(T var1, T var2) {  
          return var1 + var2;  
      }  
  };  
    
  int main() {  
      Add<int> integer;
      std::cout << "integer sum result : " << integer.Sum(3, 5) << std::endl;
    
      Add<float> floating;
      std::cout << "integer sum result : " << floating.Sum(8.8, 0.5) << std::endl;
      
      // 아래는 에러가 난다
      Add<char*> strings;
      std::cout << "string add result : " << strings.Sum("안녕", "하세요") << std::endl;
      return 0;  
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex12.png)

  이는 문자열이 입력된 경우, 문자열은 더하기를 할 수없어서 생기는 문제이다. 이런 상황을 개선하기 위해 템플릿 특수화를 통해 문자열이 입력된 경우에 다른 작업을 수행하도록 한다.

  ```cpp
  #include <iostream>
  #include <cstring>
  
  template<typename T>  
  class Add {  
  public:  
      T Sum(T var1, T var2) {  
          return var1 + var2;  
      }  
  };
  
  // const char* 타입용 특수화 함수  
  template<> 
  const char* Add<const char*>::Sum(const char* var1, const char* var2) {  
      char *sz = new char[128];  
      memset(sz, 0, sizeof(sz));  
      strcat(sz, var1);  
      strcat(sz, var2);  
      return sz;  
  }  
    
  int main() {  
      Add<int> integer;
      std::cout << "integer sum result : " << integer.Sum(3, 5) << std::endl;
    
      Add<float> floating;
      std::cout << "integer sum result : " << floating.Sum(8.8, 0.5) << std::endl;
      
      // 이제는 에러가 나지 않는다
      Add<const char*> strings;
      std::cout << "string add result : " << strings.Sum("안녕", "하세요") << std::endl;
      return 0;  
  }
  ```

  결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex13.png)

  

  템플릿 특수화는 1)**전체 템플릿 특수화**, 2)**부분 템플릿 특수화**로 나뉜다.

  1. 전체 템플릿 특수화 : 

     템플릿에서 선언한 템플릿 매개변수 전체에 대해 특정 데이터 타입으로 템플릿 인수를 지정한다. 위의 예시는 전체 템플릿 특수화다.

  2. 부분 템플릿 특수화 : 

     템플릿에서 선언한 템플릿 매개변수 가운데 일부에 대해 특정 데이터 타입으로 템플릿 인수를 지정한다.

     예시 : 

     ```cpp
     #include <iostream>
     #include <string>
     
     #define __PRETTY_FUNCTION__ __FUNCTION__ 
     
     // 1. 기본 클래스 템플릿
     template<typename Key, typename Value = std::string>
     class KeyValuePair {
     public:
         // __PRETTY_FUNCTION__ 호출한 함수와 타입을 출력하는 매크로
         KeyValuePair(Key k, Value v) : key(k), value(v) {
             std::cout << "1. " << __PRETTY_FUNCTION__ << " : " << "original" << std::endl;
             std::cout << "Key : " << key << ", Value : " << value << "\n" << std::endl;
         }
     private:
         Key key;
         Value value;
     };
     
     // 2. 위의 템플릿 인수 가운데 Value 인수에 대해 템플릿 특수화 진행
     template<typename Key>
     class KeyValuePair<Key, std::string> {
     public:
         KeyValuePair(Key k, std::string v) : key(k), value(v) {
             std::cout << "2. " << __PRETTY_FUNCTION__ << " : " << "specialized" << std::endl;;
             std::cout << "Key : " << key << ", Value : " << value << "\n" << std::endl;
         }
     private:
         Key key;
         std::string value;
     };
     
     // 3. 전체 템플릿 특수화 진행
     template<>
     class KeyValuePair<double, double> {
     public:
         KeyValuePair(double k, double v) : key(k), value(v) {
             std::cout << "3. " << __PRETTY_FUNCTION__ << " : " << " explicit full specialized" << std::endl;
             std::cout << "Key : " << key << ". Value : " << value << ", Key + Value : " << key + value << "\n" <<std::endl;
         }
     private:
         double key;
         double value;
     };
     
     int main() {
         // 1.의 클래스 템플릿 생성자 호출
         KeyValuePair<int, int> kvp1(100, 500);
         
         // 2.의 클래스 템플릿 생성자 호출
         // Value에 std::string형태의 자료가 들어오면 2.의 클래스가 호출되는 것
         KeyValuePair<int> kvp2(100, "템플릿 특수화");
         
         // 3.의 클래스 템플릿 생성자 호출
         KeyValuePair<double, double> kvp3(5.89, 6.87);
     }
     ```

     결과 : 

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/c%2B%2B/ch08/ex14.png)

  

