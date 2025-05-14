#include <vector>
#include <iostream>
using namespace std;
class Solution {
    public:
    
        double getmedian(vector<int>& nums, int len) {
            return (len%2==0) ? nums[len/2] : (nums[len/2]+nums[len/2+1])/2;
        }
    
        double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
            
            int final =nums1.size()-1+nums2.size()-1;
            vector<int> nums3(final);
    
            int num1p = 0;
            int num2p = 0;
    
            while (num1p+num2p<final) {
                int number1 = nums1[num1p];
                int number2 = nums2[num2p];
    
                if (number1<number2) {
                    nums3[num1p+num2p] = number1;
                    num1p++;
                } else {
                    nums3[num1p+num2p] = number2;
                    num2p++;
                }
    
            }
            cout<<nums3[0]<<", "<<nums3[1]<<", "<<nums3[2]<<", ";
    
            return getmedian(nums3,final);
        }
    };


int main() {
    vector<int> nums1(2); nums1[0]=1;nums1[1]=3;
    vector<int> nums2(2); nums2[0]=2;

    Solution sol = Solution();

    sol.findMedianSortedArrays(nums1,nums2);
}