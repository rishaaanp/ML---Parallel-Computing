#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <chrono>

using namespace std;

long long global_sum = 0;
bool found = false;
int key;

// Sequential Sum
long long sequential_sum(vector<int>& arr)
{
    long long sum = 0;

    for(int i = 0; i < arr.size(); i++)
        sum += arr[i];

    return sum;
}

// Sequential Search
bool sequential_search(vector<int>& arr, int key)
{
    for(int i = 0; i < arr.size(); i++)
    {
        if(arr[i] == key)
            return true;
    }

    return false;
}

// Thread Sum
void thread_sum(vector<int>& arr, int start, int end)
{
    long long local_sum = 0;

    for(int i = start; i < end; i++)
        local_sum += arr[i];

    global_sum += local_sum;
}

// Thread Search
void thread_search(vector<int>& arr, int start, int end)
{
    for(int i = start; i < end; i++)
    {
        if(arr[i] == key)
        {
            found = true;
            return;
        }
    }
}

int main()
{
    int n;

    cout << "Enter array size: ";
    cin >> n;

    vector<int> arr(n);

    // Generate random array
    for(int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    cout << "Enter key element to search: ";
    cin >> key;

    // Sequential Execution
    auto start = chrono::high_resolution_clock::now();

    long long seq_sum = sequential_sum(arr);
    bool seq_search = sequential_search(arr, key);

    auto end = chrono::high_resolution_clock::now();

    auto seq_time = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "\nSequential Results\n";
    cout << "Sum = " << seq_sum << endl;
    cout << "Key Found = " << (seq_search ? "Yes" : "No") << endl;
    cout << "Execution Time = " << seq_time.count() << " microseconds\n";

    // Thread-Based Execution
    int mid = n / 2;

    thread t1(thread_sum, ref(arr), 0, mid);
    thread t2(thread_sum, ref(arr), mid, n);

    thread t3(thread_search, ref(arr), 0, mid);
    thread t4(thread_search, ref(arr), mid, n);

    start = chrono::high_resolution_clock::now();

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    end = chrono::high_resolution_clock::now();

    auto par_time = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "\nThread-Based Results\n";
    cout << "Sum = " << global_sum << endl;
    cout << "Key Found = " << (found ? "Yes" : "No") << endl;
    cout << "Execution Time = " << par_time.count() << " microseconds\n";

    return 0;
}