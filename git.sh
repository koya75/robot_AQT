#!/bin/bash

git add 3Dmodels/*
git add agent/*
git add docker/Dockerfile
git add docker/requirements.txt
git add docker/patch/*
git add envs/*
git add image/*
git add urdf/*
git add repos/*

git commit -m "update"

git push