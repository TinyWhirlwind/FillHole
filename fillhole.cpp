#include<iostream>
#include<cmath>
#include<vector>
#include<Eigen/Eigen>
#include<Eigen/SparseLU>
#include<OpenMesh\Core\IO\MeshIO.hh>
#include<OpenMesh\Core\Mesh\TriMesh_ArrayKernelT.hh>
#include<OpenMesh\Core\Mesh\PolyMesh_ArrayKernelT.hh>
using namespace OpenMesh;
using namespace std;
using namespace Eigen;
typedef TriMesh_ArrayKernelT<> MyMesh;

//基于径向基函数的隐式曲面（RBF）
void RbfFunction(const int _originVerNum, int meshSize, MyMesh& _mesh)
{
	if (!_mesh.has_face_normals()) 
	{
		_mesh.request_face_normals();
		_mesh.update_face_normals();
	}

	//内外部控制点数
	std::vector<MyMesh::Point> outPtsList;
	MyMesh::Point outerPts;
	for (MyMesh::VertexIter v_it = _mesh.vertices_begin(); v_it != _mesh.vertices_end(); ++v_it)
	{
		if (v_it->is_boundary())
		{
			//auto verNor = _mesh.calc_vertex_normal(*v_it);
			outerPts = _mesh.point(*v_it) + _mesh.calc_vertex_normal(*v_it) * 1.0;
			outPtsList.push_back(outerPts);
		}
	}
	int ctrPtsNum = outPtsList.size();

	//稀疏矩阵
	SparseMatrix <double> A(_originVerNum + ctrPtsNum + 4, _originVerNum + ctrPtsNum + 4);
	SparseVector <double> F(_originVerNum + ctrPtsNum + 4);
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	double val;
	for (int i = _originVerNum; i < meshSize; i++)
	{
		for (int j = _originVerNum - 1; j > i; j--)
		{	
			val = pow((_mesh.point(_mesh.vertex_handle(i)) - _mesh.point(_mesh.vertex_handle(j))).length(), 3);
			tripletList.emplace_back(T(i, j, val));
			tripletList.emplace_back(T(j, i, val));
		}
	}
	//对角线
	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		for (int j = _originVerNum; j < _originVerNum + ctrPtsNum; j++)
		{
			val = pow((outPtsList[i] - _mesh.point(_mesh.vertex_handle(j))).length(), 3);
			tripletList.emplace_back(T(i, j, val));
			tripletList.emplace_back(T(j, i, val));
		}
	}

	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		for (int j = _originVerNum + ctrPtsNum; j < _originVerNum + ctrPtsNum + 3; j++)
		{
			val = _mesh.point(_mesh.vertex_handle(i))[j - _originVerNum + ctrPtsNum];
			tripletList.emplace_back(T(i, j, val));
			tripletList.emplace_back(T(j, i, val));
		}
	}

	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		tripletList.emplace_back(T(i, _originVerNum + ctrPtsNum + 3, 1.0));
		tripletList.emplace_back(T(_originVerNum + ctrPtsNum + 3, i, 1.0));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());//初始化系数矩阵
	for (int i = _originVerNum + ctrPtsNum; i < _originVerNum + ctrPtsNum; i++)
	{
		F.insert(i, 1) = 1.0;
	}
	//QR分解
	SparseLU<SparseMatrix<double>> resultMat;
	resultMat.compute(A);//对 A进行预分解
	if (resultMat.info() != Success)
	{
		return;
	}
	VectorXd vals = resultMat.solve(F);
	for (int i = _originVerNum + ctrPtsNum; i < _originVerNum + ctrPtsNum + 4; i++)
	{
		cout << "resultMat(i, 1):" << vals(i,1) << endl;
	}
	/*
	Eigen::MatrixXd  matA = MatrixXd::Zero(_originVerNum + ctrPtsNum + 4, _originVerNum + ctrPtsNum + 4);
	Eigen::MatrixXd matF = MatrixXd::Zero(_originVerNum + ctrPtsNum + 4, 1);
	Eigen::MatrixXd resultMat;
	for (int i = 0; i < _originVerNum; i++)
	{
		for (int j = _originVerNum - 1; j > i; j--)
		{
			cout << "end" << endl;
			matA(i, j) = pow((_mesh.point(_mesh.vertex_handle(i)) - _mesh.point(_mesh.vertex_handle(j))).length(), 3);
			matA(j, i) = matA(i, j);
		}
	}
	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		for (int j = _originVerNum; j < _originVerNum + ctrPtsNum; j++)
		{
			matA(i, j) = pow((outPtsList[i] - _mesh.point(_mesh.vertex_handle(j))).length(), 3);
			matA(j, i) = matA(i, j);
		}
	}

	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		for (int j = _originVerNum + ctrPtsNum; j < _originVerNum + ctrPtsNum + 3; j++)
		{
			matA(i, j) = _mesh.point(_mesh.vertex_handle(i))[j - _originVerNum + ctrPtsNum];
			matA(j, i) = matA(i, j);
		}
	}

	for (int i = 0; i < _originVerNum + ctrPtsNum; i++)
	{
		matA(i, _originVerNum + ctrPtsNum + 3) = 1.0;
		matA(_originVerNum + ctrPtsNum + 3, i) = 1.0;
	}
	for (int i = _originVerNum + ctrPtsNum; i < _originVerNum + ctrPtsNum; i++)
	{
		matF(i, 1) = 1.0;
	}
	resultMat = matA.inverse() * matF;
	for (int i = _originVerNum + ctrPtsNum; i < _originVerNum + ctrPtsNum + 4; i++)
	{
		cout <<"resultMat(i, 1):" << resultMat(i, 1) << endl;
	}*/
}

int main()
{
	MyMesh mesh;
	const char* filein = "C:\\Users\\dell\\Desktop\\homework\\out.stl";
	const char* fileout = "C:\\Users\\dell\\Desktop\\homework\\out.obj";
	std::vector<MyMesh::VertexHandle> boundaryV_list;
	if (!OpenMesh::IO::read_mesh(mesh, filein))
	{
		std::cerr << "Error:Cannot read mesh from file" << filein << std::endl;
		return 1;
	}
	else
	{
		if (mesh.faces_empty())
		{
			std::cerr << "Error:mesh file no exist faces" << filein << std::endl;
			return 1;
		}
		else
		{
			int num;
			float min_theta;
			MyMesh::Point v0;
			MyMesh::Point v1;
			MyMesh::Point v_new;
			MyMesh::Point v2;
			float b_dis;
			float theta;
			MyMesh::Point theta_nor;
			MyMesh::Point dir_pre;
			MyMesh::Point dir_next;
			int min_index;
			MyMesh::Point v_pre;
			MyMesh::Point v_next;
			MyMesh::Point half_nor;
			std::vector<MyMesh::VertexHandle> vertex_vhandle;
			MyMesh::VertexHandle new_vertex;
			std::vector<MyMesh::VertexHandle> add_vertex_list;
			int originVerNum = mesh.n_vertices();
			int originFaceNum = mesh.n_faces();
			do {
				double avg_edge_len = 0;
				num = 0;
				min_index = -1;
				theta = FLT_MAX;
				b_dis = FLT_MAX;
				min_theta = FLT_MAX;
				std::vector<MyMesh::HalfedgeHandle> half_list;
				for (MyMesh::HalfedgeIter h_it = mesh.halfedges_begin(); h_it != mesh.halfedges_end(); ++h_it)
				{
					if (h_it->is_boundary())
					{
						MyMesh::HalfedgeHandle h = *h_it;
						do {
							half_list.push_back(h);
							const MyMesh::Point& vTo = mesh.point(mesh.to_vertex_handle(h));
							const MyMesh::Point& vFrom = mesh.point(mesh.from_vertex_handle(h));
							avg_edge_len += (vFrom - vTo).length();
							num++;
							h = mesh.next_halfedge_handle(h);
							//bpoints_list.push_back(mesh.point(mesh.to_vertex_handle(h)));
						} while (h != *h_it);
						break;
					}
				}
				//std::cout << "mesh border half num is:" << num << "\n";
				avg_edge_len = avg_edge_len / num;

				for (int i = 0; i < half_list.size(); i++)
				{
					//第i+1个角的角度
					v2 = mesh.point(mesh.to_vertex_handle(half_list[(i + 1) % half_list.size()]));
					v1 = mesh.point(mesh.to_vertex_handle(half_list[i]));
					v0 = mesh.point(mesh.from_vertex_handle(half_list[i]));

					half_nor = mesh.calc_face_normal(mesh.face_handle(mesh.opposite_halfedge_handle(half_list[i])));
					dir_pre = (v1 - v0).normalized();
					dir_next = (v2 - v1).normalized();
					theta = acos((dir_pre | dir_next) / (dir_pre.length() * dir_next.length()));
					//theta_s = asin((dir_next % dir_pre).length() / (dir_pre.length() * dir_next.length()));
					theta_nor = dir_pre % dir_next;
					if ((half_nor | theta_nor) >= 0)
					{
						theta = M_PI - theta;
					}
					else
					{
						theta = M_PI + theta;
					}
					//std::cout << "u8vrgfiouechsjap-oerjhfeiodfhe" << theta << "\n";
					if (theta < min_theta)
					{
						min_theta = theta;
						min_index = i;
						b_dis = (v0 - v2).length();
						v_next = v2;
						v_pre = v0;
					}
				}
				if (b_dis < 2 * avg_edge_len)
				{
					vertex_vhandle.clear();
					vertex_vhandle.push_back(mesh.from_vertex_handle(half_list[min_index]));
					vertex_vhandle.push_back(mesh.to_vertex_handle(half_list[min_index]));
					vertex_vhandle.push_back(mesh.to_vertex_handle(half_list[(min_index + 1) % half_list.size()]));
					mesh.add_face(vertex_vhandle);
				}
				else
				{
					v_new = (v_pre + v_next) * 0.5;
					new_vertex = mesh.add_vertex(v_new);
					//新增点集
					add_vertex_list.push_back(new_vertex);
					vertex_vhandle.clear();
					vertex_vhandle.push_back(mesh.from_vertex_handle(half_list[min_index]));
					vertex_vhandle.push_back(mesh.to_vertex_handle(half_list[min_index]));
					vertex_vhandle.push_back(new_vertex);
					mesh.add_face(vertex_vhandle);
					vertex_vhandle.clear();
					vertex_vhandle.push_back(mesh.to_vertex_handle(half_list[min_index]));
					vertex_vhandle.push_back(mesh.to_vertex_handle(half_list[(min_index + 1) % half_list.size()]));
					vertex_vhandle.push_back(new_vertex);
					mesh.add_face(vertex_vhandle);
				}
				//std::cout << "faces num is:" << mesh.n_faces()<< "\n";
			} while (num>3);
			OpenMesh::IO::write_mesh(mesh, fileout);
			std::cout << "faces num is:" << mesh.n_vertices() << "\n";
			std::cout << "faces num is:" << mesh.n_faces() << "\n";
			//网格优化
			int matSize = mesh.n_vertices();
			double weightPts;
			Eigen::MatrixXd matL = MatrixXd::Zero(matSize, matSize);
			Eigen::MatrixXd matX = MatrixXd::Zero(matSize, 3);
			Eigen::MatrixXd resultMat = MatrixXd::Zero(matSize, 3);
			for (MyMesh::VertexIter v_it = mesh.vertices_sbegin(); v_it != mesh.vertices_end(); ++v_it)
			{
				matL(v_it->idx(), v_it->idx()) = 1.0;
				if (v_it->idx() >= originVerNum || v_it->is_boundary())
				{
					matL(v_it->idx(), v_it->idx()) = 0.0;
					weightPts = 1.0 / v_it->valence();
					//weightPts = -1.0 / v_it->valence();
					for (MyMesh::VertexVertexIter vv_it = mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it)
					{
						matL(v_it->idx(), vv_it->idx()) = weightPts;
						//std::cout << "mesh.point:(" << mesh.point(*vv_it)[0] << mesh.point(*vv_it)[1] << mesh.point(*vv_it)[2] << ")" << "\n";
					}
				}
				matX(v_it->idx(), 0) = mesh.point(*v_it)[0];
				matX(v_it->idx(), 1) = mesh.point(*v_it)[1];
				matX(v_it->idx(), 2) = mesh.point(*v_it)[2];
				std::cout << "matX:(" << matX(v_it->idx(), 0) << matX(v_it->idx(), 1) << matX(v_it->idx(), 2) << ")" << "\n";
			}
			resultMat = matL/*.inverse() */* matX;
			for (MyMesh::VertexIter v_it = mesh.vertices_sbegin(); v_it != mesh.vertices_end(); ++v_it)
			{
				std::cout << "resultMat:(" << resultMat(v_it->idx(), 0) << resultMat(v_it->idx(), 1)<< resultMat(v_it->idx(), 2) <<")" << "\n";
				if (v_it->idx() >= originVerNum || v_it->is_boundary())
				{
					mesh.point(*v_it)[0] = resultMat(v_it->idx(), 0);
					mesh.point(*v_it)[1] = resultMat(v_it->idx(), 1);
					mesh.point(*v_it)[2] = resultMat(v_it->idx(), 2);
				}
				std::cout << "mesh.point:(" << mesh.point(*v_it)[0] << mesh.point(*v_it)[1] << mesh.point(*v_it)[2] << ")" << "\n";
			}
			OpenMesh::IO::write_mesh(mesh, "C:\\Users\\dell\\Desktop\\homework\\out1.obj");
			//径向基
			RbfFunction(originVerNum, matSize, mesh);
		}
	}
}

