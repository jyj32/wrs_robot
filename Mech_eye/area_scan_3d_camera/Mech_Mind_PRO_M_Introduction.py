"""
Author: Yixuan Su
Date: 2024/12/15 11:04
File: Mech_Mind_PRO_M_Introduction.py
Description:
"""

"""
    例程分为以下类别：basic、advanced 和 util。    
    basic 例程：连接和采集数据。    
    advanced 例程：通过复杂、高阶的方式采集数据，设置部分型号特有参数。    
    util 例程：获取相机信息和设置通用参数。    
    各分类中包含的例程及其简介如下。    
    
    basic
    connect_to_camera：连接相机。    
    connect_and_capture_images：连接相机并获取2D图、深度图及点云数据。    
    capture_2d_image：从相机获取并保存2D图。    
    capture_depth_map：从相机获取并保存深度图。    
    capture_point_cloud：从相机获取并保存无纹理点云和纹理点云。    
    capture_point_cloud_hdr：设置多个曝光时间，然后从相机获取并保存点云。    
    capture_point_cloud_with_normals：计算法向量，并保存含法向量的点云。
        
    advanced
    convert_depth_map_to_point_cloud：从深度图生成并保存点云。    
    multiple_cameras_capture_sequentially：使用多台相机按序获取并保存2D图、深度图及点云。    
    multiple_cameras_capture_simultaneously：使用多台相机同时获取并保存2D图、深度图及点云。    
    capture_periodically：在设定时间内，定时获取并保存2D图、深度图和点云。    
    mapping_2d_image_to_depth_map：从覆盖掩膜的2D图和深度图生成并保存无纹理点云和纹理点云。    
    set_parameters_of_laser_cameras：设置激光相机特有的参数。    
    set_parameters_of_uhp_cameras：设置UHP系列相机特有的参数。    
    register_camera_event：定义并注册检测相机连接状态的回调函数。    
    capture_stereo_2d_images：获取Deep（V3）、Laser L Enhanced（V3）、PRO XS（V4）、LSR L（V4）、LSR S（V4）和DEEP（V4）的两个2D相机的2D图像。
    
    util
    get_camera_intrinsics：获取并打印相机内参。    
    print_camera_info：获取并打印相机型号、序列号、固件版本、温度等信息。    
    set_scanning_parameters：设置3D参数、2D参数和感兴趣区域分组下的参数。    
    set_depth_range：设置深度范围参数。    
    set_point_cloud_processing_parameters：设置点云后处理参数。    
    manage_user_sets：管理参数组，如获取所有参数组的名称、新增参数组、切换参数组和保存参数设置至参数组。    
    save_and_load_user_set：从JSON文件导入并替换所有参数组，将所有参数组保存为JSON文件。
    
"""
