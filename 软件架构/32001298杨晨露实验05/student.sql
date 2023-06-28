/*
Navicat MySQL Data Transfer

Source Server         : zhongxia
Source Server Version : 80013
Source Host           : localhost:3306
Source Database       : suzy

Target Server Type    : MYSQL
Target Server Version : 80013
File Encoding         : 65001

Date: 2020-06-06 13:59:06
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for student
-- ----------------------------
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `stuNo` int(11) NOT NULL AUTO_INCREMENT,
  `stuName` varchar(255) DEFAULT NULL,
  `stuAge` int(11) DEFAULT NULL,
  PRIMARY KEY (`stuNo`)
) ENGINE=InnoDB AUTO_INCREMENT=112 DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- ----------------------------
-- Records of student
-- ----------------------------
INSERT INTO `student` VALUES ('1', 'yuner', '29');
INSERT INTO `student` VALUES ('2', 'suzy', '25');
INSERT INTO `student` VALUES ('3', 'lebron', '35');
INSERT INTO `student` VALUES ('10', 'lyf', '32');
INSERT INTO `student` VALUES ('19', 'library', '21');
INSERT INTO `student` VALUES ('23', 'zhongxia', '21');
INSERT INTO `student` VALUES ('77', '77', '77');
