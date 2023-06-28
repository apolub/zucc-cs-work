package cn.edu.zucc.cache.service;

import cn.edu.zucc.cache.entity.User;

/**
 * @author longzhonghua
 * @data 2019/01/28 17:47
 */

public interface UserService {
    public User findUserById(long id);
    public User insertUser(User user);
    public User updateUserById(User user);
    public void deleteUserById(long id);
}
